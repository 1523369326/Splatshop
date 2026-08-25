#pragma once

#include "InputAction.h"

// Measures the Euclidean distance between two positions picked from the splat surface.
struct TwoPointMeasureAction : public InputAction{

	vector<vec3> points;
	bool hasHoveredPoint = false;
	vec3 hoveredPoint = {0.0f, 0.0f, 0.0f};

	float pointMarkerSize(vec3 point){
		vec3 cameraPosition = vec3{Runtime::controls->getPosition()};
		float cameraDistance = glm::length(cameraPosition - point);
		return 14.0f * cameraDistance / float(GLRenderer::width);
	}

	void drawMarker(SplatEditor* editor, vec3 point, vec4 color){
		editor->drawSphere({
			.pos = point,
			.scale = vec3{pointMarkerSize(point)},
			.color = color,
		});
	}

	void update(){
		auto editor = SplatEditor::instance;
		bool isLeftClicked = Runtime::mouseEvents.button == 0 && Runtime::mouseEvents.action == 1;

		hasHoveredPoint = editor->deviceState.hovered_depth != Infinity;
		if(hasHoveredPoint){
			hoveredPoint = editor->deviceState.hovered_pos;
		}

		if(isLeftClicked && hasHoveredPoint && points.size() < 2){
			points.push_back(hoveredPoint);
		}

		for(vec3 point : points){
			drawMarker(editor, point, {1.0f, 0.85f, 0.0f, 1.0f});
		}

		if(points.empty()) return;

		vec3 end = points.size() == 2 ? points[1] : hoveredPoint;
		if(points.size() == 1 && !hasHoveredPoint) return;

		editor->drawLine(points[0], end, 0xff00d8ff);
		if(points.size() == 1 && hasHoveredPoint){
			drawMarker(editor, hoveredPoint, {0.0f, 0.85f, 1.0f, 0.8f});
		}
	}

	void makeToolbarSettings(){
		ImGui::TextUnformatted("Two-point measurement");
		ImGui::SameLine();

		if(points.empty()){
			ImGui::TextUnformatted("Click the first point.");
			return;
		}

		if(points.size() == 1){
			if(hasHoveredPoint){
				float distance = glm::length(hoveredPoint - points[0]);
				ImGui::Text("Preview: %.4f world units. Click the second point.", distance);
			}else{
				ImGui::TextUnformatted("Move over a splat, then click the second point.");
			}
			return;
		}

		float distance = glm::length(points[1] - points[0]);
		ImGui::Text("Distance: %.4f world units.", distance);
		ImGui::SameLine();
		ImGui::TextUnformatted("Right-click to close; choose Measure again for a new measurement.");
	}
};
