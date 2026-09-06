
void SplatEditor::makeSettings(){

	auto editor = SplatEditor::instance;
	auto& settings = editor->settings;
	Scene& scene = editor->scene;

	// SETTINGS
	if (ImGui::CollapsingHeader("Settings", ImGuiTreeNodeFlags_DefaultOpen))
	{ 

		if(ImGui::Button("Toggle Inset")){
			settings.showInset = !settings.showInset;
		}
		ImGui::SliderFloat("SplatSize", &settings.splatSize, 0.0, 2.0);

		ImGui::Checkbox("Solid", &settings.showSolid);
		ImGui::SameLine();
		ImGui::Checkbox("Tiles", &settings.showTiles);
		ImGui::SameLine();
		ImGui::Checkbox("Show Ring", &settings.showRing);
		ImGui::SameLine();
		ImGui::Checkbox("Point", &settings.makePoints);

		ImGui::Text("Mode: "); ImGui::SameLine();
		ImGui::RadioButton("Color##renermode", &settings.rendermode, RENDERMODE_COLOR); ImGui::SameLine();
		ImGui::RadioButton("Depth##renermode", &settings.rendermode, RENDERMODE_DEPTH); ImGui::SameLine();
		ImGui::RadioButton("Tiles##renermode", &settings.rendermode, RENDERMODE_TILES); ImGui::SameLine();
		ImGui::RadioButton("Heatmap##renermode", &settings.rendermode, RENDERMODE_HEATMAP); ImGui::SameLine();

		if(ImGui::Checkbox("RaDe-GS geometry", &settings.enableRaDeGeometry)){
			settings.showRaDeGeometryPanel = settings.enableRaDeGeometry;
			if(!settings.enableRaDeGeometry && settings.rendermode == RENDERMODE_NORMAL)
				settings.rendermode = RENDERMODE_COLOR;
		}
		if(settings.enableRaDeGeometry){
			ImGui::TextUnformatted("RaDe-GS output:"); ImGui::SameLine();
			ImGui::RadioButton("RGB##rade-settings", &settings.rendermode, RENDERMODE_COLOR); ImGui::SameLine();
			ImGui::RadioButton("Depth##rade-settings", &settings.rendermode, RENDERMODE_DEPTH); ImGui::SameLine();
			ImGui::RadioButton("Normal##rade-settings", &settings.rendermode, RENDERMODE_NORMAL);
		}

		// ImGui::SameLine();
		if(ImGui::Button("Save Cubins")){
			writeBinaryFile(format("./{}.cubin", "gaussians") , (uint8_t*)editor->prog_gaussians_editing->cubin, editor->prog_gaussians_editing->cubinSize);
			writeBinaryFile(format("./{}.cubin", "gaussians") , (uint8_t*)editor->prog_gaussians_editing->cubin, editor->prog_gaussians_editing->cubinSize);
			writeBinaryFile(format("./{}.cubin", "points")    , (uint8_t*)editor->prog_points->cubin,    editor->prog_points->cubinSize);
			writeBinaryFile(format("./{}.cubin", "triangles") , (uint8_t*)editor->prog_triangles->cubin, editor->prog_triangles->cubinSize);
			writeBinaryFile(format("./{}.cubin", "lines")     , (uint8_t*)editor->prog_lines->cubin,     editor->prog_lines->cubinSize);
			writeBinaryFile(format("./{}.cubin", "helpers")   , (uint8_t*)editor->prog_helpers->cubin,   editor->prog_helpers->cubinSize);
		}

		// ImGui::SameLine();
		// if(ImGui::Button("Save Scene WIP")){
		// 	// SplatsyWriter::write("./test.splatsy", scene, *Runtime::controls);
		// 	SplatsyFilesWriter::write("./savefile", scene, *Runtime::controls);
		// }

	}
}
