#pragma once

#ifdef SPLATSHOP_ENABLE_GAUSSIAN_SELECTOR
#include "actions/GaussianSelectorAction.h"
#endif

// Each feature owns its expandable controls. Add future research/tools here
// without growing the viewport toolbar.
void SplatEditor::makeDevFeaturesGUI() {
	if(!settings.showDevFeatures) return;
	auto editor = this;
	ImGui::SetNextWindowPos(ImVec2(500, 104), ImGuiCond_FirstUseEver);
	ImGui::SetNextWindowSize(ImVec2(490, 650), ImGuiCond_FirstUseEver);
	if(!ImGui::Begin("Dev", &settings.showDevFeatures)) {
		ImGui::End();
		return;
	}
	ImGui::TextWrapped("Research reproductions and practical tools");
	if(ImGui::TreeNodeEx("Camera Studio", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::TextWrapped("Camera paths, pose editing, independent preview and video recording.");
		if(ImGui::Button("Open Camera Studio")) {
			cameraStudio->open(*this);
			if(cameraStudio->isOpen()) settings.showDevFeatures = false;
		}
		ImGui::TreePop();
	}
	if(ImGui::TreeNodeEx("GaussianSelector", ImGuiTreeNodeFlags_DefaultOpen)) {
#ifdef SPLATSHOP_ENABLE_GAUSSIAN_SELECTOR
		auto action=std::dynamic_pointer_cast<GaussianSelectorAction>(editor->state.currentAction);
		if(!action && ImGui::Button("Start GaussianSelector")) {
			action=std::make_shared<GaussianSelectorAction>();editor->setAction(action);
		}
		if(action) action->makePanel();
		else ImGui::TextWrapped("Select a Gaussian layer, then start to draw an ROI and foreground/background strokes.");
#else
		ImGui::TextWrapped("Not enabled in this build. Configure SPLATSHOP_ENABLE_GAUSSIAN_SELECTOR=ON to use this tool.");
#endif
		ImGui::TreePop();
	}
	ImGui::End();
}
