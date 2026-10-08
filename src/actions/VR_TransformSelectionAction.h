
#pragma once

#include "InputAction.h"

struct VR_TransformSelectionUndoAction : public Action{

	// // Action either affects a single node, or all nodes if node == nullptr
	// SceneNode* node = nullptr; 

	// // forward transformation. Use inverse for undo.
	// mat4 transform;

	// If numSelectedSplats == 0: Transform selected node
	// If numSelectedSplats >  0: Transform selected splats
	int64_t numSelectedSplats = 0;
	unordered_map<SNSplats*, mat4> transforms;

	void undo(){
		auto editor = SplatEditor::instance;

		for(auto [node, transform] : transforms){

			if(numSelectedSplats == 0){
				node->transform = inverse(transform) * node->transform;
			}else{
				mat4 t = inverse(node->transform) * transform * node->transform;
				editor->applyTransformation(node->dmng.data, inverse(t), true);
			}
		}
	}

	void redo(){
		auto editor = SplatEditor::instance;

		for(auto [node, transform] : transforms){

			if(numSelectedSplats == 0){
				node->transform = transform * node->transform;
			}else{
				mat4 t = inverse(node->transform) * transform * node->transform;
				editor->applyTransformation(node->dmng.data, t, true);
			}
		}
	}

	int64_t byteSize(){
		return 0; // Not enough to matter
	}
};

struct VR_TransformSelectionAction : public InputAction{

	shared_ptr<VR_TransformSelectionUndoAction> currentUndoAction = nullptr;
	// mat4 cummulative = mat4(1.0f);

	~VR_TransformSelectionAction(){
		stop();
	}

	void beginUndoable(){
		currentUndoAction = make_shared<VR_TransformSelectionUndoAction>();
		currentUndoAction->numSelectedSplats = SplatEditor::instance->getNumSelectedSplats();
	}

	void endUndoable(){
		auto editor = SplatEditor::instance;

		if(currentUndoAction){
			editor->addAction(currentUndoAction);
			currentUndoAction = nullptr;
		}
	}

	void start(){
		println("starting VR_TransformSelectionAction");
	}

	void update(){
		auto editor = SplatEditor::instance;
		auto& settings = editor->settings;
		Scene& scene = editor->scene;
		auto ovr = editor->ovr;

		if(!ovr->isActive()) return;

		Controller right = ovr->getRightController();

		if(right.isBPressed() && !right.wasBPressed()){
			beginUndoable();
			// cummulative = mat4(1.0f);
		}else if(!right.isBPressed() && right.wasBPressed()){
			endUndoable();
		}

		if(right.isBPressed() && right.wasBPressed()){

			dmat4 world = scene.world->transform;
			mat4 delta = (inverse(world) * ovr->flip * right.transform) * inverse((inverse(world) * ovr->flip * right.previous.transform));

			if(currentUndoAction->numSelectedSplats == 0){
				// Transform selected node
				shared_ptr<SceneNode> selectedNode = editor->getSelectedNode();

				onTypeMatch<SNSplats>(selectedNode, [=](shared_ptr<SNSplats> node){
					node->transform = delta * node->transform;

					if(!currentUndoAction->transforms.contains(node.get())){
						currentUndoAction->transforms[node.get()] = mat4(1.0f);
					}
					currentUndoAction->transforms[node.get()] = delta * currentUndoAction->transforms[node.get()];
				});
			}else{
				// Transform selected splats
				scene.process<SNSplats>([&](SNSplats* node){
					bool onlySelected = true;

					mat4 t = inverse(node->transform) * delta * (node->transform);

					editor->applyTransformation(node->dmng.data,t,onlySelected);

					if(!currentUndoAction->transforms.contains(node)){
						currentUndoAction->transforms[node] = mat4(1.0f);
					}

					currentUndoAction->transforms[node] = delta * currentUndoAction->transforms[node];
				});
			}
		}
	}

	void stop(){
		auto editor = SplatEditor::instance;

		println("stopping VR_TransformSelectionAction");
		endUndoable();
	}

};
