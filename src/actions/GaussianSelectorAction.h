#pragma once

#ifdef SPLATSHOP_ENABLE_GAUSSIAN_SELECTOR

#include "InputAction.h"
#include "GaussianSelectorRuntime.h"

#include <unordered_map>

struct GaussianSelectorAction : InputAction {
	enum class StrokeMode { Foreground, Background, Erase };
	struct PixelChange { uint32_t index; uint8_t beforeFg, beforeBg, afterFg, afterBg; };
	struct Segment { ImVec2 a, b; StrokeMode mode; float radius; };
	struct StrokeDelta { std::vector<PixelChange> changes; std::vector<Segment> segments; };
	struct Canvas {
		bool hasRoi = false;
		ImVec2 roiStart{}, roiEnd{};
		int width = 0, height = 0;
		std::vector<uint8_t> foreground, background;
		std::vector<StrokeDelta> undo, redo;
		std::vector<Segment> segments;
	};

	StrokeMode strokeMode = StrokeMode::Foreground;
	gaussian_selector::ApplyMode applyMode = gaussian_selector::ApplyMode::Replace;
	float brushSize = 20.0f;
	std::unordered_map<uint64_t, Canvas> canvases;
	uint64_t currentView = 0;
	bool drawing = false;
	bool drawRoi = true;
	bool draggingRoi = false;
	bool applyPending = false;
	uint64_t applyRequest = 0;
	std::string hint;
	bool modelHistory = false;
	std::weak_ptr<SNSplats> requestTarget;
	ImVec2 previous{};
	std::unordered_map<uint32_t, uint8_t> strokeBefore;
	std::size_t strokeSegmentBegin = 0;

	void start() override {
		auto* editor = SplatEditor::instance;
		if(!editor->gaussianSelector) editor->gaussianSelector = std::make_shared<GaussianSelectorRuntime>(editor);
		editor->settings.showDevFeatures=true;
	}

	Canvas& canvas() {
		auto& c = canvases[currentView];
		const int width = GLRenderer::width, height = GLRenderer::height;
		if(c.width != width || c.height != height) {
			c = {};
			c.width = width; c.height = height;
			c.foreground.assign(std::size_t(width) * height, 0);
			c.background.assign(std::size_t(width) * height, 0);
		}
		return c;
	}

	void paintDisk(Canvas& c, int cx, int cy) {
		const int radius = std::max(1, int(std::ceil(brushSize)));
		for(int y = std::max(0, cy-radius); y <= std::min(c.height-1, cy+radius); ++y)
		for(int x = std::max(0, cx-radius); x <= std::min(c.width-1, cx+radius); ++x) {
			if(!c.hasRoi || x<std::min(c.roiStart.x,c.roiEnd.x) || x>std::max(c.roiStart.x,c.roiEnd.x) ||
				y<std::min(c.roiStart.y,c.roiEnd.y) || y>std::max(c.roiStart.y,c.roiEnd.y)) continue;
			const int dx=x-cx, dy=y-cy; if(dx*dx+dy*dy > radius*radius) continue;
			const uint32_t i = uint32_t(x + y*c.width);
			if(!strokeBefore.contains(i)) strokeBefore[i] = uint8_t((c.foreground[i]?1:0) | (c.background[i]?2:0));
			if(strokeMode == StrokeMode::Foreground) { c.foreground[i]=1; c.background[i]=0; }
			else if(strokeMode == StrokeMode::Background) { c.foreground[i]=0; c.background[i]=1; }
			else { c.foreground[i]=0; c.background[i]=0; }
		}
	}

	void paintLine(Canvas& c, ImVec2 a, ImVec2 b) {
		const float dx=b.x-a.x, dy=b.y-a.y;
		const int steps=std::max(1, int(std::ceil(std::sqrt(dx*dx+dy*dy)/std::max(1.0f,brushSize*.35f))));
		for(int s=0;s<=steps;++s){const float t=float(s)/steps;paintDisk(c,int(std::round(a.x+t*dx)),int(std::round(a.y+t*dy)));}
		c.segments.push_back({a,b,strokeMode,brushSize});
	}

	void finishStroke(Canvas& c) {
		StrokeDelta delta;
		for(const auto& [i, before] : strokeBefore) {
			const uint8_t after=uint8_t((c.foreground[i]?1:0)|(c.background[i]?2:0));
			if(before!=after) delta.changes.push_back({i,uint8_t(before&1),uint8_t((before>>1)&1),c.foreground[i],c.background[i]});
		}
		if(strokeSegmentBegin<c.segments.size())delta.segments.assign(c.segments.begin()+strokeSegmentBegin,c.segments.end());
		if(!delta.changes.empty()){c.undo.push_back(std::move(delta));c.redo.clear();}
		strokeBefore.clear(); drawing=false;
	}

	bool undoInteractionStep() override {
		if(modelHistory) return false;
		invalidate();
		if(!canvases.contains(currentView)) return false; auto& c=canvas(); if(c.undo.empty()) return false;
		auto d=std::move(c.undo.back());c.undo.pop_back();for(const auto& p:d.changes){c.foreground[p.index]=p.beforeFg;c.background[p.index]=p.beforeBg;}if(d.segments.size()<=c.segments.size())c.segments.resize(c.segments.size()-d.segments.size());c.redo.push_back(std::move(d));return true;
	}
	bool redoInteractionStep() override {
		if(modelHistory) return false;
		invalidate();
		if(!canvases.contains(currentView)) return false; auto& c=canvas(); if(c.redo.empty()) return false;
		auto d=std::move(c.redo.back());c.redo.pop_back();for(const auto& p:d.changes){c.foreground[p.index]=p.afterFg;c.background[p.index]=p.afterBg;}c.segments.insert(c.segments.end(),d.segments.begin(),d.segments.end());c.undo.push_back(std::move(d));return true;
	}

	void update() override {
		auto* editor=SplatEditor::instance;editor->gaussianSelector->poll();
		const auto key=editor->gaussianSelector->evidenceViewKey(GLRenderer::width,GLRenderer::height,mat4(GLRenderer::camera->view),GLRenderer::camera->proj);
		if(key!=currentView) {
			if(drawing && canvases.contains(currentView)) finishStroke(canvases[currentView]);
			draggingRoi=false; invalidate(); currentView=key;
		}
		if(applyPending && editor->gaussianSelector->requestId()==applyRequest && editor->gaussianSelector->hasPreview()) {
			if(requestTarget.lock()!=selectedTarget()) {invalidate();hint="Target layer changed; run Apply again.";}
			else if(editor->gaussianSelector->apply(applyMode)) { editor->gaussianSelector->resetPreview(); modelHistory=true; }
			applyPending=false;
		}
		auto& c=canvas();
		const ImVec2 p{Runtime::mouseEvents.pos_x,Runtime::mouseEvents.pos_y};
		const bool blocked=ImGui::GetIO().WantCaptureMouse;
		if(!blocked && Runtime::mouseEvents.isLeftDownEvent()) {
			invalidate(); modelHistory=false;
			if(drawRoi) {
				c.foreground.assign(c.foreground.size(),0); c.background.assign(c.background.size(),0);
				c.undo.clear(); c.redo.clear(); c.segments.clear();
				editor->gaussianSelector->clearAllEvidence();
				c.roiStart=c.roiEnd={std::clamp(p.x,0.0f,float(c.width-1)),std::clamp(p.y,0.0f,float(c.height-1))}; c.hasRoi=false; draggingRoi=true;
			} else if(c.hasRoi) {drawing=true;previous=p;strokeBefore.clear();strokeSegmentBegin=c.segments.size();paintLine(c,p,p);}
		}
		if(draggingRoi) {
			c.roiEnd={std::clamp(p.x,0.0f,float(c.width-1)),std::clamp(p.y,0.0f,float(c.height-1))};
			if(Runtime::mouseEvents.isLeftUpEvent()) {
				draggingRoi=false; c.hasRoi=std::abs(c.roiEnd.x-c.roiStart.x)>=4 && std::abs(c.roiEnd.y-c.roiStart.y)>=4;
				if(c.hasRoi) {drawRoi=false;strokeMode=StrokeMode::Foreground;}
			}
		}
		else if(drawing && Runtime::mouseEvents.isLeftDown && !blocked){paintLine(c,previous,p);previous=p;}
		if(drawing && Runtime::mouseEvents.isLeftUpEvent())finishStroke(c);

		auto* draw=ImGui::GetForegroundDrawList();
		if(c.hasRoi || draggingRoi) {
			ImVec2 lo{std::min(c.roiStart.x,c.roiEnd.x),float(c.height)-std::max(c.roiStart.y,c.roiEnd.y)};
			ImVec2 hi{std::max(c.roiStart.x,c.roiEnd.x),float(c.height)-std::min(c.roiStart.y,c.roiEnd.y)};
			draw->AddRect(lo,hi,IM_COL32(255,200,40,255),0,0,2);
			draw->PushClipRect(lo,hi,true);
			// Draw the actual masks: erased evidence disappears instead of leaving gray strokes.
			for(int y=int(std::min(c.roiStart.y,c.roiEnd.y)); y<=int(std::max(c.roiStart.y,c.roiEnd.y)); ++y) {
				for(int x=int(std::min(c.roiStart.x,c.roiEnd.x)); x<=int(std::max(c.roiStart.x,c.roiEnd.x));) {
					const auto i=std::size_t(y)*c.width+x;
					const int label=c.foreground[i]?1:(c.background[i]?2:0); const int begin=x++;
					while(x<=int(std::max(c.roiStart.x,c.roiEnd.x))) {const auto j=std::size_t(y)*c.width+x; if((c.foreground[j]?1:(c.background[j]?2:0))!=label)break; ++x;}
					if(label)draw->AddRectFilled({float(begin),float(c.height-y-1)},{float(x),float(c.height-y)},label==1?IM_COL32(30,255,60,180):IM_COL32(255,45,45,180));
				}
			}
			draw->PopClipRect();
		}
		const ImU32 cursor=strokeMode==StrokeMode::Foreground?IM_COL32(30,255,60,255):(strokeMode==StrokeMode::Background?IM_COL32(255,45,45,255):IM_COL32(220,220,220,255));
		if(!blocked && !drawRoi) draw->AddCircle({p.x,float(c.height)-p.y},brushSize,cursor,48,2.0f);
	}

	void invalidate() {
		applyPending=false;
		if(auto runtime=SplatEditor::instance->gaussianSelector) { runtime->cancel(); runtime->resetPreview(); }
	}

	void solve(bool applyAfter) {
		auto runtime=SplatEditor::instance->gaussianSelector; auto& c=canvas();
		if(!c.hasRoi) {hint="Draw an ROI rectangle first.";return;}
		if(!supportedRenderer()) {hint="Use standard single-view 3DGS; disable Solid, Core-Tail, RaDe and renderer variants.";return;}
		hint.clear();
		requestTarget=selectedTarget();
		glm::vec4 box{std::min(c.roiStart.x,c.roiEnd.x),std::min(c.roiStart.y,c.roiEnd.y),std::max(c.roiStart.x,c.roiEnd.x),std::max(c.roiStart.y,c.roiEnd.y)};
		runtime->requestSolve(selectedTarget(),c.width,c.height,mat4(GLRenderer::camera->view),GLRenderer::camera->proj,c.foreground,c.background,box);
		applyPending=applyAfter; applyRequest=runtime->requestId();
	}

	std::shared_ptr<SNSplats> selectedTarget() const {
		auto selected=SplatEditor::instance->getSelectedNode();
		return std::dynamic_pointer_cast<SNSplats>(selected);
	}

	bool supportedRenderer() const {
		const auto& s=SplatEditor::instance->settings;
		return s.splatRenderer==SPLATRENDERER_3DGS && !s.renderSoA && !s.renderBandwidth && !s.renderFragIntersections && !s.enableRaDeGeometry && !s.multiView
			&& s.rendermode!=RENDERMODE_HEATMAP && s.rendermode!=RENDERMODE_CORETAIL
			&& !s.showSolid && !s.enableSplatCulling;
	}

	void makeToolbarSettings() override {
		ImGui::TextUnformatted("GaussianSelector");
		ImGui::SameLine();
		if(ImGui::Button("Controls")) SplatEditor::instance->settings.showDevFeatures=true;
		ImGui::SameLine();
		ImGui::TextUnformatted("Dev > GaussianSelector");
	}

	void makePanel() {
		auto runtime=SplatEditor::instance->gaussianSelector;
		ImGui::TextWrapped("1. Draw ROI  2. Paint foreground/background  3. Apply selection");
		if(ImGui::Button("Draw / redraw ROI")) { invalidate(); drawRoi=true; }
		ImGui::SameLine();
		ImGui::TextUnformatted(drawRoi ? "Drag in the viewport" : "ROI ready");
		int mode=int(strokeMode);
		if(ImGui::RadioButton("Foreground",&mode,0)) drawRoi=false;
		ImGui::SameLine(); if(ImGui::RadioButton("Background",&mode,1)) drawRoi=false;
		ImGui::SameLine(); if(ImGui::RadioButton("Erase",&mode,2)) drawRoi=false;
		strokeMode=StrokeMode(mode);
		ImGui::SliderFloat("Brush radius",&brushSize,2,100,"%.0f px");
		if(ImGui::Button("Undo stroke")) {modelHistory=false;undoInteractionStep();}
		ImGui::SameLine();if(ImGui::Button("Redo stroke")) {modelHistory=false;redoInteractionStep();}
		ImGui::SameLine();if(ImGui::Button("Clear strokes")) {
			auto& c=canvas(); std::fill(c.foreground.begin(),c.foreground.end(),0);std::fill(c.background.begin(),c.background.end(),0);
			c.undo.clear();c.redo.clear();c.segments.clear();invalidate();runtime->clearAllEvidence();
		}
		int operation=int(applyMode);
		if(ImGui::Combo("Selection operation",&operation,"Replace inside ROI\0Add\0Subtract\0Intersect inside ROI\0")) {applyMode=gaussian_selector::ApplyMode(operation);invalidate();}
		if(ImGui::Button("Preview")) solve(false);
		ImGui::SameLine();if(ImGui::Button("Apply selection")) solve(true);
		ImGui::SameLine();if(ImGui::Button("Cancel")) invalidate();
		ImGui::TextWrapped("Apply creates an ordinary selection for move, copy, delete and export. Outside ROI is unchanged. Ctrl+Z/Y undo/redo Apply after it finishes.");
		{
			auto& cfg=runtime->config();
			int budget=int(cfg.graphNodeBudget);
			if(ImGui::SliderInt("Superpoint node budget",&budget,2000,100000)) {cfg.graphNodeBudget=budget;cfg.graphNodeTarget=uint32_t(.8*budget);invalidate();}
			if(ImGui::Checkbox("Keep foreground-seeded components",&cfg.keepSeededComponents))invalidate();
			if(ImGui::Checkbox("Reject oversized Gaussians",&cfg.rejectOversizedGaussians))invalidate();
			if(ImGui::Button("Use author-notes profile")) {
				const auto old=cfg;cfg=gaussian_selector::SelectorConfig{};
				cfg.spatialWeight=old.spatialWeight;cfg.colorWeight=old.colorWeight;cfg.opacityWeight=old.opacityWeight;
				cfg.cpuBudgetBytes=old.cpuBudgetBytes;cfg.gpuBudgetBytes=old.gpuBudgetBytes;
				runtime->clearAllEvidence();invalidate();
			}
		}
		if(!hint.empty()) ImGui::TextWrapped("%s",hint.c_str());
		ImGui::TextWrapped("%s: %s",gaussianSelectorStateName(runtime->state()),runtime->message().c_str());
		ImGui::Text("Scope %zu / %zu | SP %zu | edges %zu",runtime->scopedGaussianCount(),runtime->gaussianCount(),runtime->superpointCount(),runtime->edgeCount());
		ImGui::Text("Encoding nodes %zu (%s)",runtime->encodingNodeCount(),runtime->voxelPooled()?"voxel pooled":"exact Gaussian graph");
		if(runtime->hasPreview()) {
			const auto& d=runtime->diagnostics();
			ImGui::Text("FG/BG seeds %zu / %zu | split conflicts %zu / %zu",d.foregroundSeeds,d.backgroundSeeds,d.splitSuperpoints,d.conflictingSuperpoints);
			ImGui::Text("Uncertain splits %zu | removed components %zu | oversized %zu",d.uncertainSplits,d.removedUnseededComponents,d.removedOversizedGaussians);
			ImGui::Text("Solver: %s | EM <= %u | kNN %u",d.integerCut?"integer push-relabel":"float baseline",runtime->config().emIterations,runtime->config().softKnn);
			if(d.pseudoBackgroundSamples)ImGui::TextWrapped("Using %zu weak pseudo-BG samples (engineering ranking). Draw real BG strokes for a more reliable contrast model.",d.pseudoBackgroundSamples);
			if(d.degradedAppearance) ImGui::TextWrapped("Appearance model has limited evidence; add foreground/background strokes on representative surfaces.");
		}
		if(ImGui::TreeNode("Algorithm parameters")) {
			auto& cfg=runtime->config(); bool changed=false;
			float resolution=float(cfg.leidenResolution),lambda=float(cfg.pairwiseLambda),coverage=float(cfg.coverageThreshold);
			if(ImGui::SliderFloat("Leiden resolution",&resolution,.001f,.3f,"%.3f")) {cfg.leidenResolution=resolution;changed=true;}
			if(ImGui::SliderFloat("Pairwise weight",&lambda,0,10)) {cfg.pairwiseLambda=lambda;changed=true;}
			if(ImGui::SliderFloat("Coverage threshold",&coverage,.01f,1)) {cfg.coverageThreshold=coverage;runtime->clearAllEvidence();changed=true;}
			if(ImGui::SliderFloat("Evidence transmittance stop",&cfg.evidenceTransmittanceStop,0,.95f,"%.2f")) {runtime->clearAllEvidence();changed=true;}
			if(ImGui::Button("Standard replay (0)")) {cfg.evidenceTransmittanceStop=0;cfg.evidenceOwnerBudget=0;runtime->clearAllEvidence();changed=true;}
			ImGui::SameLine();if(ImGui::Button("T = 0.5")) {cfg.evidenceTransmittanceStop=.5f;cfg.evidenceOwnerBudget=16;runtime->clearAllEvidence();changed=true;}
			ImGui::SameLine();if(ImGui::Button("Notes T = 0.9")) {cfg.evidenceTransmittanceStop=.9f;cfg.evidenceOwnerBudget=16;runtime->clearAllEvidence();changed=true;}
			ImGui::Text("Evidence owner budget: %u (0 = unlimited)",cfg.evidenceOwnerBudget);
			ImGui::TextWrapped("Notes profile: T=0.9, at most 16 owners per pixel. Higher T stop gives shallower evidence. FG/BG and full-view mass use the same visibility rule.");
			if(ImGui::Checkbox("Split FG/BG superpoint conflicts",&cfg.splitConflictingSuperpoints)) changed=true;
			if(ImGui::Checkbox("Post-cut uncertain SP refinement",&cfg.refineUncertainSuperpoints))changed=true;
			if(ImGui::Checkbox("GMM + weighted kNN contrast",&cfg.useSoftKnnContrast))changed=true;
			float mix=float(cfg.unaryGmmMix);if(ImGui::SliderFloat("GMM mix (engineering value)",&mix,0,1)){cfg.unaryGmmMix=mix;changed=true;}
			if(ImGui::Checkbox("Weak pseudo-BG without BG strokes",&cfg.allowPseudoBackground))changed=true;
			if(ImGui::Checkbox("Integer graph cut",&cfg.integerGraphCut))changed=true;
			if(ImGui::Checkbox("Adaptive coverage Q10",&cfg.adaptiveCoverageThreshold)){runtime->clearAllEvidence();changed=true;}
			if(ImGui::Checkbox("Require visible scribble center (engineering gate)",&cfg.requireCenterEvidence)){runtime->clearAllEvidence();changed=true;}
			float scaleFactor=float(cfg.oversizedMedianMultiplier);if(ImGui::SliderFloat("Oversized scale factor (engineering)",&scaleFactor,2,32)){cfg.oversizedMedianMultiplier=scaleFactor;changed=true;}
			ImGui::Text("Graph weights: spatial %.2f | CAC %.2f | opacity %.2f",cfg.spatialWeight,cfg.colorWeight,cfg.opacityWeight);
			bool restricted=cfg.refinementMode==gaussian_selector::RefinementMode::ForegroundRestrictedExperimental;
			if(ImGui::Checkbox("Experimental FG refinement",&restricted)) {cfg.refinementMode=restricted?gaussian_selector::RefinementMode::ForegroundRestrictedExperimental:gaussian_selector::RefinementMode::InteractiveFullGraph;changed=true;}
			if(changed)invalidate();
			ImGui::TextWrapped("ROI uses projected Gaussian centers. All scene occluders contribute to transmittance; the denominator includes every pixel under the selected evidence visibility rule.");
			ImGui::TreePop();
		}
	}
	void stop() override {
		if(auto runtime=SplatEditor::instance->gaussianSelector){runtime->cancel();runtime->resetPreview();}
		drawing=false;strokeBefore.clear();
	}
};

#endif
