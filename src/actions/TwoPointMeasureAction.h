#pragma once

#include "InputAction.h"

// Measures the Euclidean distance between two positions picked from the splat surface.
// The CDF/PDF mode freezes one camera ray at a time and lets the user choose among
// multiple alpha-contribution peaks before committing the point.
struct TwoPointMeasureAction : public InputAction{

	static constexpr int PDF_BIN_COUNT = 128;

	vector<vec3> points;
	bool hasHoveredPoint = false;
	vec3 hoveredPoint = {0.0f, 0.0f, 0.0f};

	bool profileFrozen = false;
	bool profileCapturePending = false;
	vector<float> pdf;
	vector<float> cdf;
	vector<int> peakBins;
	float profileDepthMin = 0.0f;
	float profileDepthMax = 0.0f;
	float selectedDepth = 0.0f;
	float totalProfileWeight = 0.0f;
	int selectedBin = 0;
	uint32_t capturedSampleCount = 0;
	bool capturedProfileOverflow = false;
	int frozenPixelX = 0;
	int frozenPixelY = 0;
	int frozenWidth = 1;
	int frozenHeight = 1;
	mat4 frozenView = mat4(1.0f);
	mat4 frozenProj = mat4(1.0f);
	mat4 requestedView = mat4(1.0f);
	mat4 requestedProj = mat4(1.0f);
	int requestedPixelX = 0;
	int requestedPixelY = 0;
	int requestedWidth = 1;
	int requestedHeight = 1;

	bool projectToScreen(vec3 point, ImVec2& screenPosition){
		vec4 clipPosition = mat4(GLRenderer::camera->proj) * mat4(GLRenderer::camera->view) * vec4(point, 1.0f);
		if(clipPosition.w <= 0.0f) return false;

		vec3 ndc = vec3(clipPosition) / clipPosition.w;
		screenPosition = {
			(ndc.x * 0.5f + 0.5f) * GLRenderer::width,
			(1.0f - (ndc.y * 0.5f + 0.5f)) * GLRenderer::height,
		};
		return true;
	}

	void drawMarker(vec3 point, ImU32 color){
		ImVec2 screenPosition;
		if(projectToScreen(point, screenPosition)){
			ImGui::GetBackgroundDrawList()->AddCircleFilled(screenPosition, 7.0f, color);
		}
	}

	vec3 reconstructFrozenPoint(float depth){
		vec4 viewPosition = {
			((2.0f * float(frozenPixelX) / float(frozenWidth) - 1.0f) * depth) / frozenProj[0][0],
			((2.0f * float(frozenPixelY) / float(frozenHeight) - 1.0f) * depth) / frozenProj[1][1],
			-depth,
			1.0f,
		};
		return vec3(glm::inverse(frozenView) * viewPosition);
	}

	void updateSelectedDepth(){
		if(pdf.empty()) return;
		selectedBin = clamp(selectedBin, 0, int(pdf.size()) - 1);
		float u = (float(selectedBin) + 0.5f) / float(pdf.size());
		selectedDepth = profileDepthMin + u * (profileDepthMax - profileDepthMin);
		hoveredPoint = reconstructFrozenPoint(selectedDepth);
		hasHoveredPoint = true;
	}

	void beginProfileCapture(SplatEditor* editor){
		requestedPixelX = int(Runtime::mouseEvents.pos_x);
		requestedPixelY = int(Runtime::mouseEvents.pos_y);
		requestedWidth = max(GLRenderer::width, 1);
		requestedHeight = max(GLRenderer::height, 1);
		requestedView = mat4(GLRenderer::camera->view);
		requestedProj = mat4(GLRenderer::camera->proj);

		editor->settings.measurementProfilePixelX = requestedPixelX;
		editor->settings.measurementProfilePixelY = requestedPixelY;
		editor->settings.measurementProfilePixelLocked = true;
		profileCapturePending = true;
		hasHoveredPoint = false;
	}

	void cancelProfileCapture(SplatEditor* editor){
		editor->settings.measurementProfilePixelLocked = false;
		profileCapturePending = false;
	}

	bool captureProfile(SplatEditor* editor){
		auto& source = editor->deviceState.measurementProfile;
		if(!profileCapturePending || source.pixelX != requestedPixelX || source.pixelY != requestedPixelY){
			return false;
		}
		uint32_t sampleCount = min(source.count, MEASUREMENT_PROFILE_MAX_SAMPLES);
		vector<MeasurementDepthSample> samples;
		samples.reserve(sampleCount);

		for(uint32_t i = 0; i < sampleCount; i++){
			auto sample = source.samples[i];
			if(std::isfinite(sample.depth) && sample.depth > 0.0f && std::isfinite(sample.weight) && sample.weight > 0.0f){
				samples.push_back(sample);
			}
		}
		if(samples.empty()) return false;

		std::sort(samples.begin(), samples.end(), [](const auto& a, const auto& b){
			return a.depth < b.depth;
		});

		profileDepthMin = samples.front().depth;
		profileDepthMax = samples.back().depth;
		if(profileDepthMax - profileDepthMin < 1.0e-5f){
			float padding = max(profileDepthMin * 0.005f, 1.0e-3f);
			profileDepthMin -= padding;
			profileDepthMax += padding;
		}else{
			float padding = (profileDepthMax - profileDepthMin) * 0.02f;
			profileDepthMin = max(0.0f, profileDepthMin - padding);
			profileDepthMax += padding;
		}

		vector<float> histogram(PDF_BIN_COUNT, 0.0f);
		totalProfileWeight = 0.0f;
		float depthRange = profileDepthMax - profileDepthMin;
		for(auto sample : samples){
			float normalizedDepth = (sample.depth - profileDepthMin) / depthRange;
			int bin = clamp(int(normalizedDepth * PDF_BIN_COUNT), 0, PDF_BIN_COUNT - 1);
			histogram[bin] += sample.weight;
			totalProfileWeight += sample.weight;
		}

		cdf.assign(PDF_BIN_COUNT, 0.0f);
		float cumulativeWeight = 0.0f;
		for(int i = 0; i < PDF_BIN_COUNT; i++){
			cumulativeWeight += histogram[i];
			cdf[i] = totalProfileWeight > 0.0f ? cumulativeWeight / totalProfileWeight : 0.0f;
		}

		float sigma = max(editor->settings.measurementPdfBandwidthBins, 0.5f);
		int radius = int(ceil(3.0f * sigma));
		pdf.assign(PDF_BIN_COUNT, 0.0f);
		for(int bin = 0; bin < PDF_BIN_COUNT; bin++){
			float weightedValue = 0.0f;
			float kernelWeight = 0.0f;
			for(int offset = -radius; offset <= radius; offset++){
				int sourceBin = bin + offset;
				if(sourceBin < 0 || sourceBin >= PDF_BIN_COUNT) continue;
				float kernel = exp(-0.5f * float(offset * offset) / (sigma * sigma));
				weightedValue += histogram[sourceBin] * kernel;
				kernelWeight += kernel;
			}
			pdf[bin] = kernelWeight > 0.0f ? weightedValue / kernelWeight : 0.0f;
		}

		float maxPdf = 0.0f;
		int strongestBin = 0;
		for(int i = 0; i < PDF_BIN_COUNT; i++){
			if(pdf[i] > maxPdf){
				maxPdf = pdf[i];
				strongestBin = i;
			}
		}
		if(maxPdf <= 0.0f) return false;
		for(float& value : pdf) value /= maxPdf;

		vector<int> peakCandidates;
		for(int i = 0; i < PDF_BIN_COUNT; i++){
			float left = i > 0 ? pdf[i - 1] : -1.0f;
			float right = i + 1 < PDF_BIN_COUNT ? pdf[i + 1] : -1.0f;
			if(pdf[i] >= left && pdf[i] > right && pdf[i] >= 0.015f){
				peakCandidates.push_back(i);
			}
		}
		std::sort(peakCandidates.begin(), peakCandidates.end(), [&](int a, int b){
			return pdf[a] > pdf[b];
		});

		peakBins.clear();
		int minimumPeakDistance = max(2, int(ceil(2.0f * sigma)));
		for(int candidate : peakCandidates){
			bool separated = true;
			for(int accepted : peakBins){
				if(abs(candidate - accepted) < minimumPeakDistance){
					separated = false;
					break;
				}
			}
			if(separated) peakBins.push_back(candidate);
			if(peakBins.size() == 8) break;
		}
		if(peakBins.empty()) peakBins.push_back(strongestBin);
		std::sort(peakBins.begin(), peakBins.end());

		capturedSampleCount = uint32_t(samples.size());
		capturedProfileOverflow = source.overflow != 0 || source.count > MEASUREMENT_PROFILE_MAX_SAMPLES;
		frozenPixelX = requestedPixelX;
		frozenPixelY = requestedPixelY;
		frozenWidth = requestedWidth;
		frozenHeight = requestedHeight;
		frozenView = requestedView;
		frozenProj = requestedProj;
		selectedBin = strongestBin;
		cancelProfileCapture(editor);
		profileFrozen = true;
		updateSelectedDepth();
		return true;
	}

	void commitSelectedPoint(){
		if(!profileFrozen || !hasHoveredPoint) return;
		points.push_back(hoveredPoint);
		profileFrozen = false;
		pdf.clear();
		cdf.clear();
		peakBins.clear();

		if(points.size() == 2){
			auto editor = SplatEditor::instance;
			editor->measurements.push_back({.start = points[0], .end = points[1]});
			editor->setAction(nullptr);
		}
	}

	void stop() override{
		SplatEditor::instance->settings.measurementProfilePixelLocked = false;
		profileCapturePending = false;
	}

	void drawMeasurementPreview(){
		for(vec3 point : points){
			drawMarker(point, IM_COL32(255, 216, 0, 255));
		}
		if(points.empty()){
			if(profileFrozen && hasHoveredPoint){
				drawMarker(hoveredPoint, IM_COL32(0, 216, 255, 204));
			}
			return;
		}
		if(!hasHoveredPoint) return;

		ImVec2 startScreen;
		ImVec2 endScreen;
		if(projectToScreen(points[0], startScreen) && projectToScreen(hoveredPoint, endScreen)){
			ImGui::GetBackgroundDrawList()->AddLine(startScreen, endScreen, IM_COL32(255, 216, 0, 255), 1.5f);
		}
		drawMarker(hoveredPoint, IM_COL32(0, 216, 255, 204));
	}

	void update(){
		auto editor = SplatEditor::instance;
		bool isLeftClicked = Runtime::mouseEvents.button == 0 && Runtime::mouseEvents.action == 1;

		if(editor->settings.measurementDepthMode == 3){
			if(profileFrozen){
				updateSelectedDepth();
			}else if(profileCapturePending){
				captureProfile(editor);
			}else{
				hasHoveredPoint = editor->deviceState.hovered_depth != Infinity;
				if(hasHoveredPoint) hoveredPoint = editor->deviceState.hovered_pos;
				if(isLeftClicked && hasHoveredPoint) beginProfileCapture(editor);
			}
			drawMeasurementPreview();
			return;
		}

		hasHoveredPoint = editor->deviceState.hovered_depth != Infinity;
		if(hasHoveredPoint){
			hoveredPoint = editor->deviceState.hovered_pos;
		}

		if(isLeftClicked && hasHoveredPoint && points.size() < 2){
			points.push_back(hoveredPoint);

			if(points.size() == 2){
				editor->measurements.push_back({.start = points[0], .end = points[1]});
				editor->setAction(nullptr);
				return;
			}
		}

		drawMeasurementPreview();
	}

	void drawPdfSelector(){
		if(pdf.empty()) return;

		float availableWidth = ImGui::GetContentRegionAvail().x;
		ImVec2 graphSize = {min(max(availableWidth, 320.0f), 760.0f), 150.0f};
		ImVec2 graphMin = ImGui::GetCursorScreenPos();
		ImGui::InvisibleButton("PDF depth selector", graphSize, ImGuiButtonFlags_MouseButtonLeft);
		ImVec2 graphMax = {graphMin.x + graphSize.x, graphMin.y + graphSize.y};
		ImDrawList* drawList = ImGui::GetWindowDrawList();
		drawList->AddRectFilled(graphMin, graphMax, IM_COL32(18, 20, 24, 255), 4.0f);
		drawList->AddRect(graphMin, graphMax, IM_COL32(90, 96, 108, 255), 4.0f);

		if(ImGui::IsItemActive() && ImGui::IsMouseDown(0)){
			float u = clamp((ImGui::GetIO().MousePos.x - graphMin.x) / graphSize.x, 0.0f, 1.0f);
			selectedBin = clamp(int(u * PDF_BIN_COUNT), 0, PDF_BIN_COUNT - 1);
			updateSelectedDepth();
		}

		for(int i = 1; i < PDF_BIN_COUNT; i++){
			float x0 = graphMin.x + graphSize.x * float(i - 1) / float(PDF_BIN_COUNT - 1);
			float x1 = graphMin.x + graphSize.x * float(i) / float(PDF_BIN_COUNT - 1);
			float pdfY0 = graphMax.y - 8.0f - pdf[i - 1] * (graphSize.y - 16.0f);
			float pdfY1 = graphMax.y - 8.0f - pdf[i] * (graphSize.y - 16.0f);
			float cdfY0 = graphMax.y - 8.0f - cdf[i - 1] * (graphSize.y - 16.0f);
			float cdfY1 = graphMax.y - 8.0f - cdf[i] * (graphSize.y - 16.0f);
			drawList->AddLine({x0, pdfY0}, {x1, pdfY1}, IM_COL32(255, 205, 40, 255), 2.0f);
			drawList->AddLine({x0, cdfY0}, {x1, cdfY1}, IM_COL32(70, 170, 255, 180), 1.0f);
		}

		for(int peakBin : peakBins){
			float x = graphMin.x + graphSize.x * (float(peakBin) + 0.5f) / float(PDF_BIN_COUNT);
			float y = graphMax.y - 8.0f - pdf[peakBin] * (graphSize.y - 16.0f);
			drawList->AddCircleFilled({x, y}, 4.0f, IM_COL32(255, 120, 40, 255));
		}

		float selectedX = graphMin.x + graphSize.x * (float(selectedBin) + 0.5f) / float(PDF_BIN_COUNT);
		drawList->AddLine({selectedX, graphMin.y}, {selectedX, graphMax.y}, IM_COL32(80, 255, 150, 255), 2.0f);

		if(ImGui::IsItemHovered()){
			float u = clamp((ImGui::GetIO().MousePos.x - graphMin.x) / graphSize.x, 0.0f, 1.0f);
			float hoveredDepth = profileDepthMin + u * (profileDepthMax - profileDepthMin);
			ImGui::SetTooltip("Depth %.6f\nDrag to move the selected 3D point", hoveredDepth);
		}
	}

	void makeToolbarSettings(){
		if(SplatEditor::instance->settings.measurementDepthMode != 3){
			ImGui::TextUnformatted("Two-point measurement");
			ImGui::SameLine();

			if(points.empty()){
				ImGui::TextUnformatted("Click the first point.");
			}else if(hasHoveredPoint){
				float distance = glm::length(hoveredPoint - points[0]);
				ImGui::Text("Preview: %.4f world units. Click the second point.", distance);
			}else{
				ImGui::TextUnformatted("Move over a splat, then click the second point.");
			}
			return;
		}

		ImGui::TextUnformatted("CDF/PDF multi-surface picking");
		if(!profileFrozen){
			ImGui::SameLine();
			if(profileCapturePending){
				ImGui::TextUnformatted("Sampling the clicked ray on the GPU...");
			}else{
				ImGui::TextUnformatted(points.empty()
					? "Click the first ray to sample its depth distribution."
					: "Click the second ray to sample its depth distribution.");
			}
			uint32_t liveCount = min(SplatEditor::instance->deviceState.measurementProfile.count, MEASUREMENT_PROFILE_MAX_SAMPLES);
			ImGui::Text("Live Gaussian samples: %u", liveCount);
			if(profileCapturePending && ImGui::Button("Cancel sampling")){
				cancelProfileCapture(SplatEditor::instance);
			}
			return;
		}

		ImGui::Text("PDF (yellow), CDF (blue), detected peaks (orange), selected depth (green)");
		drawPdfSelector();
		ImGui::Text("Selected depth: %.6f | probability mass: %.3f | samples: %u%s",
			selectedDepth, totalProfileWeight, capturedSampleCount, capturedProfileOverflow ? " (truncated)" : "");
		ImGui::TextUnformatted("Detected surfaces:");
		for(int i = 0; i < int(peakBins.size()); i++){
			if(i > 0) ImGui::SameLine();
			int peakBin = peakBins[i];
			float peakU = (float(peakBin) + 0.5f) / float(PDF_BIN_COUNT);
			float peakDepth = profileDepthMin + peakU * (profileDepthMax - profileDepthMin);
			ImGui::PushID(i);
			if(ImGui::SmallButton(format("Peak %d: %.4f", i + 1, peakDepth).c_str())){
				selectedBin = peakBin;
				updateSelectedDepth();
			}
			ImGui::PopID();
		}

		if(ImGui::Button(points.empty() ? "Use as first point" : "Use as second point")){
			commitSelectedPoint();
			return;
		}
		ImGui::SameLine();
		if(ImGui::Button("Resample ray")){
			cancelProfileCapture(SplatEditor::instance);
			profileFrozen = false;
			pdf.clear();
			cdf.clear();
			peakBins.clear();
		}
	}
};
