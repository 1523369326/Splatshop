
#pragma once

#include "InputAction.h"

struct VR_BrushSelectUndoAction : public Action{
	unordered_map<SNSplats*, CUdeviceptr> splatmasks;
	BRUSHMODE brushmode;

	~VR_BrushSelectUndoAction(){
		for(auto [node, cptr] : splatmasks){
			CURuntime::free(cptr);
		}
	}

	void undo(){
		auto editor = SplatEditor::instance;

		for(auto [node, mask] : splatmasks){
			void* args[] = {&node->dmng.data, &mask};

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

		for(auto [node, mask] : splatmasks){
			void* args[] = {&node->dmng.data, &mask};
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
		for(auto [node, cptr] : splatmasks){
			if(cptr)  totalSize += CURuntime::getSize(cptr);
		}

		return totalSize;
	}
};

struct VR_BrushSelectAction : public InputAction{

	shared_ptr<VR_BrushSelectUndoAction> currentUndoAction = nullptr;

	const dmat4 flip = glm::dmat4(
		1.0,  0.0, 0.0, 0.0,
		0.0,  0.0, 1.0, 0.0,
		0.0, -1.0, 0.0, 0.0,
		0.0, 0.0, 0.0, 1.0
	);

	void start(){
		println("starting VR_BrushSelectAction");

	}

	void update(){
		auto editor = SplatEditor::instance;
		auto& settings = editor->settings;
		Scene& scene = editor->scene;
		auto ovr = editor->ovr;

		if(!ovr->isActive()) return;

		Controller right = ovr->getRightController();

		if(right.triggered()){
			beginUndoable();
		}else if(right.untriggered()){
			endUndoable();
		}

		scene.process<SNSplats>([&](SNSplats* node) {
			auto data = node->dmng.data;

			if(!node->visible) return;
			if(node->locked) return;

			data.transform = node->transform_global;
			
			CUdeviceptr cptr_changedmask = 0;
			if(currentUndoAction && currentUndoAction->splatmasks.contains(node)){
				cptr_changedmask = currentUndoAction->splatmasks[node];
			}

			bool isTriggerPressed = right.isTriggerPressed();
			mat4 controllerPose = right.transform;

			void* args[] = {&editor->launchArgs, &data, &isTriggerPressed, &controllerPose, &cptr_changedmask};
			editor->prog_gaussians_editing->launch("kernel_select_vr", args, data.count);
		});

		{ // Place Brush Cage
			mat4 mTranslate = translate(vec3{0.0f, 0.1f, 0.0f});
			mat4 mScale = scale(vec3{0.1f, 0.1f, 0.1f});
			mat4 mRot = glm::rotate(-140.0f * 3.1415f / 180.0f, vec3{ 1.0f, 0.0f, 0.0f });
			mat4 transform_controller = mat4(flip * right.transform) * mRot * mTranslate * mScale;

			editor->sn_brushsphere->transform = transform_controller;
			editor->sn_brushsphere->locked = true;
			editor->sn_brushsphere->visible = true;

			// set brush color (specifically made for 4*16 bit colors)
			static_assert(sizeof(Color) == 8);
			Color value;
			if(settings.brush.mode == BRUSHMODE::SELECT){
				value = {0xffff, 0x0000, 0x0000, 0xffff};
			}else if(settings.brush.mode == BRUSHMODE::ERASE){
				value = {0x0000, 0x0000, 0xffff, 0xffff};
			}
			Color* destination = editor->sn_brushsphere->dmng.data.color;
			uint32_t count = editor->sn_brushsphere->dmng.data.count;
			
			void* args[] = {&destination, &value, &count};
			editor->prog_gaussians_editing->launch("kernel_memset_u64", args, count);
		}

	}

	void stop(){
		auto editor = SplatEditor::instance;
		println("stopping VR_BrushSelectAction");
		endUndoable();
		editor->sn_brushsphere->visible = false;
	}

	void beginUndoable(){
		auto editor = SplatEditor::instance;
		auto& settings = editor->settings;

		// Create new undo action
		currentUndoAction = make_shared<VR_BrushSelectUndoAction>();
		currentUndoAction->brushmode = settings.brush.mode;

		// Create splatmasks
		editor->scene.process<SNSplats>([&](SNSplats* node) {
			if(!node->visible) return;
			if(node->locked) return;

			auto data = node->dmng.data;
			data.transform = node->transform_global;

			uint32_t requiredBits = node->dmng.data.count;
			uint32_t requiredBytes = (requiredBits / 8) + 4;
			CUdeviceptr cptr_splatmask = CURuntime::alloc("SphereSelectUndoAction splatmask", requiredBytes);
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

};
