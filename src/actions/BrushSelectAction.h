#pragma once

#include "InputAction.h"

struct BrushSelectUndoAction : public Action{

	unordered_map<shared_ptr<SNSplats>, CUdeviceptr> splatmasks;
	BRUSHMODE brushmode;
	
	~BrushSelectUndoAction(){
		for(auto [node, cptr_splatmask] : splatmasks){
			CURuntime::free(cptr_splatmask);
		}
	}

	void undo(){
		auto editor = SplatEditor::instance;

		for(auto [node, cptr_splatmask] : splatmasks){
			void* args[] = {&node->dmng.data, &cptr_splatmask};

			if(brushmode == BRUSHMODE::SELECT){
				editor->prog_gaussians_editing->launch("kernel_deselect_masked", args, node->dmng.data.count);
			}else if(brushmode == BRUSHMODE::ERASE){
				editor->prog_gaussians_editing->launch("kernel_undelete_masked", args, node->dmng.data.count);
				node->contentRevision++;
			}
		}
	}

	void redo(){
		auto editor = SplatEditor::instance;

		for(auto [node, cptr_splatmask] : splatmasks){
			void* args[] = {&node->dmng.data, &cptr_splatmask};

			if(brushmode == BRUSHMODE::SELECT){
				editor->prog_gaussians_editing->launch("kernel_select_masked", args, node->dmng.data.count);
			}else if(brushmode == BRUSHMODE::ERASE){
				editor->prog_gaussians_editing->launch("kernel_delete_masked", args, node->dmng.data.count);
				node->contentRevision++;
			}
		}
	}

	int64_t byteSize(){
		
		int64_t totalSize = 0;
		for(auto [node, splatmask] : splatmasks){
			if(splatmask)  totalSize += CURuntime::getSize(splatmask);
		}

		return totalSize;
	}

};

struct BrushSelectAction : public InputAction{

	shared_ptr<BrushSelectUndoAction> currentUndoAction = nullptr;

	void start(){
	
	}

	void update(){

		auto editor = SplatEditor::instance;
		auto& settings = editor->settings;

		if(settings.brush.mode == BRUSHMODE::SELECT){
			ImVec2 position = {Runtime::mouseEvents.pos_x, Runtime::mouseEvents.pos_y};
			drawDashedCircle_ba(position, settings.brush.size, IM_COL32(255,   0,   0, 255));
		}else if(settings.brush.mode == BRUSHMODE::ERASE){
			ImVec2 position = {Runtime::mouseEvents.pos_x, Runtime::mouseEvents.pos_y};
			drawDashedCircle_ba(position, settings.brush.size, IM_COL32(  0,   0,   0, 255));
			//editor->state.hasPendingDeletionTask = true;
		}else if(settings.brush.mode == BRUSHMODE::REMOVE_FLAGS){
			ImVec2 position = {Runtime::mouseEvents.pos_x, Runtime::mouseEvents.pos_y};
			drawDashedCircle_ba(position, settings.brush.size, IM_COL32(  0, 255, 255, 255));
		}

		static bool wasLeftDown = false;
		bool isLeftDown = Runtime::mouseEvents.isLeftDown;
		bool leftReleased = wasLeftDown && !isLeftDown;
		bool leftPressed = wasLeftDown == false && isLeftDown == true;
		wasLeftDown = isLeftDown;

		// Each press starts a new separate undoable action
		if(leftPressed){
			beginUndoable();
		}else if(leftReleased){
			endUndoable();
		}

		RenderTarget target;
		target.width = GLRenderer::width;
		target.height = GLRenderer::height;
		target.framebuffer = (uint64_t*)editor->virt_framebuffer->cptr;
		target.indexbuffer = nullptr;
		target.view = mat4(GLRenderer::camera->view); // * scene.transform;
		target.proj = GLRenderer::camera->proj;

		if(settings.brush.mode != BRUSHMODE::NONE){
			editor->scene.process<SNSplats>([&](shared_ptr<SNSplats> node) {

				if(!node->visible) return;
				if(node->locked) return;

				auto data = node->dmng.data;
				data.transform = node->transform_global;

				CUdeviceptr cptr_splatmask = 0;
				if(currentUndoAction){
					cptr_splatmask = currentUndoAction->splatmasks[node];
				}

				void* args[] = {&editor->launchArgs, &target, &data, &cptr_splatmask};
				editor->prog_gaussians_editing->launch("kernel_brushselect", args, data.count);
			});

		}
		
	}

	void stop(){
		auto editor = SplatEditor::instance;

		endUndoable();

		editor->settings.brush.mode = BRUSHMODE::NONE; 
	}

	void makeToolbarSettings(){
		auto editor = SplatEditor::instance;

		ImGui::Text("Brush Size: "); 
		
		ImGui::SameLine();
		ImGui::SetNextItemWidth(100.0f);
		ImGui::SliderFloat(" ", &editor->settings.brush.size, 5.0f, 200.0f, "%.1f");

		ImGui::SameLine(); ImGui::Text("|"); 

		int intersectionmode = editor->settings.brush.intersectionmode;
		ImGui::SameLine();
		ImGui::Text("Brush intersects splat's"); 

		ImGui::SameLine();
		ImGui::RadioButton("Center", &intersectionmode, BRUSH_INTERSECTION_CENTER); 

		ImGui::SameLine();
		ImGui::RadioButton("Border", &intersectionmode, BRUSH_INTERSECTION_BORDER); 
		editor->settings.brush.intersectionmode = intersectionmode;

		ImGui::SameLine(); ImGui::Text("|"); 

		ImGui::SameLine(); 
		ImGui::Text("Affects splats larger than (pixels): "); 

		ImGui::SameLine();
		ImGui::SetNextItemWidth(100.0f);
		ImGui::SliderFloat(" ", &editor->settings.brush.minSplatSize, 0.0f, 1000.0f, "%.1f");
	}

	void beginUndoable(){
		auto editor = SplatEditor::instance;
		auto& settings = editor->settings;

		// Create new undo action
		currentUndoAction = make_shared<BrushSelectUndoAction>();
		currentUndoAction->brushmode = editor->settings.brush.mode;

		// Create splatmasks
		editor->scene.process<SNSplats>([&](shared_ptr<SNSplats> node) {

			if(!node->visible) return;
			if(node->locked) return;

			auto data = node->dmng.data;
			data.transform = node->transform_global;

			uint32_t requiredBits = node->dmng.data.count;
			uint32_t requiredBytes = (requiredBits / 8) + 4;
			CUdeviceptr cptr_splatmask = CURuntime::alloc("BrushAction splatmask", requiredBytes);
			cuMemsetD8(cptr_splatmask, 0, requiredBytes);
			currentUndoAction->splatmasks[node] = cptr_splatmask;
		
		});
	}

	void endUndoable(){
		auto editor = SplatEditor::instance;

		editor->scene.process<SNSplats>([&](SNSplats* node) {

			if(!node->visible) return;
			if(node->locked) return;

			auto data = node->dmng.data;
			data.transform = node->transform_global;

			editor->prog_gaussians_editing->launch("kernel_clear_highlighting", {&editor->launchArgs, &data}, data.count);
		});

		if(currentUndoAction){
			if(currentUndoAction->brushmode == BRUSHMODE::ERASE) for(auto& [node, _] : currentUndoAction->splatmasks) node->contentRevision++;
			editor->addAction(currentUndoAction);
			currentUndoAction = nullptr;
		}
	}

	void drawDashedCircle_ba(ImVec2 position, float radius, ImU32 color){
		auto drawlist = ImGui::GetForegroundDrawList();

		ImU32 white = IM_COL32(255, 255, 255, 127);

		float thickness = 2.0f;

		float width = GLRenderer::width;
		float height = GLRenderer::height;
		int steps = 100;
		for(int i = 0; i < steps; i += 2){

			float u_0 = 2.0f * 3.1415f * float(i + 0) / float(steps);
			float u_1 = 2.0f * 3.1415f * float(i + 1) / float(steps);
			float u_2 = 2.0f * 3.1415f * float(i + 2) / float(steps);

			ImVec2 start = {
				radius * cos(u_0) + position.x,
				height - (radius * sin(u_0) + position.y),
			};
			ImVec2 end = {
				radius * cos(u_1) + position.x,
				height - (radius * sin(u_1) + position.y),
			};
			ImVec2 end2 = {
				radius * cos(u_2) + position.x,
				height - (radius * sin(u_2) + position.y),
			};

			drawlist->AddLine(start, end2, white, 1.0f);
			drawlist->AddLine(start, end, color, thickness);
		}
	}

};




