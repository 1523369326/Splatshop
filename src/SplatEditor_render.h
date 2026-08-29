
void SplatEditor::render(){

	cuStreamSynchronize(0);
	cuStreamSynchronize(mainstream);

	if(GLRenderer::width * GLRenderer::height == 0){
		ImGui::Render();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
		return;
	}

	{ // adjust framebuffer size
		GLRenderer::view.framebuffer->setSize(GLRenderer::width, GLRenderer::height);

		uint64_t requiredBytes = GLRenderer::width * GLRenderer::height * 8;
		virt_framebuffer->commit(requiredBytes);
	}

	glBindFramebuffer(GL_FRAMEBUFFER, GLRenderer::view.framebuffer->handle);

	static CUevent event_render_start;
	static CUevent event_render_end;
	static bool initialized = false;
	if(!initialized){
		cuEventCreate(&event_render_start, CU_EVENT_DEFAULT);
		cuEventCreate(&event_render_end, CU_EVENT_DEFAULT);
		initialized = true;
	}
	
	if(ovr->isActive())
	{ // RENDER VR

		if(Runtime::measureTimings){
			cuEventRecord(event_render_start, 0);
		}

		auto poseHMD = ovr->getHmdPose();
		auto poseLeft = ovr->getEyePose(vr::Hmd_Eye::Eye_Left);
		auto poseRight = ovr->getEyePose(vr::Hmd_Eye::Eye_Right);

		auto size = ovr->getRecommmendedRenderTargetSize();	
		int width = size[0];
		int height = size[1];

		

		Runtime::debugValues["vr resolution"] = format("{} x {}", width, height);

		if(!settings.enableOverlapped)
		{ // LEFT

			virt_framebuffer->commit(width * height * 8);

			RenderTarget target;
			target.width = width;
			target.height = height;
			target.framebuffer = (uint64_t*)virt_framebuffer->cptr;
			target.indexbuffer = nullptr;
			target.view = mat4(viewLeft.view);// * scene.transform;
			target.proj = viewLeft.proj;
			target.VP = viewLeft.VP;

			vector<RenderTarget> targets = { target };
			draw(&scene, targets);

			Rectangle targetViewport;
			targetViewport.x = 0;
			targetViewport.y = 0;
			targetViewport.width = width;
			targetViewport.height = height;

			auto glMapping = mapCudaGl(viewLeft.framebuffer->colorAttachments[0]);
			prog_gaussians_rendering->launch("kernel_blit_opengl", 
				{&launchArgs, &target, &glMapping.surface, &targetViewport}, 
				targetViewport.width * targetViewport.height);
			glMapping.unmap();
		}

		if(!settings.enableOverlapped)
		{ // RIGHT
			virt_framebuffer->commit(width * height * 8);

			RenderTarget target;
			target.width = width;
			target.height = height;
			target.framebuffer = (uint64_t*)virt_framebuffer->cptr;
			target.indexbuffer = nullptr;
			target.view = mat4(viewRight.view); // * scene.transform;
			target.proj = viewRight.proj;
			target.VP = viewRight.VP;

			vector<RenderTarget> targets = { target };
			draw(&scene, targets);

			Rectangle targetViewport;
			targetViewport.x = 0;
			targetViewport.y = 0;
			targetViewport.width = width;
			targetViewport.height = height;

			auto glMapping = mapCudaGl(viewRight.framebuffer->colorAttachments[0]);
			prog_gaussians_rendering->launch("kernel_blit_opengl", 
				{&launchArgs, &target, &glMapping.surface, &targetViewport}, 
				targetViewport.width * targetViewport.height);
			glMapping.unmap();
		}

		if(settings.enableOverlapped)
		{ // LEFT & RIGHT
			shared_ptr<CudaVirtualMemory> virt_framebuffer_left = virt_framebuffer;
			static shared_ptr<CudaVirtualMemory> virt_framebuffer_right = CURuntime::allocVirtual("framebuffer_right");

			virt_framebuffer_left->commit(width * height * 8);
			virt_framebuffer_right->commit(width * height * 8);

			RenderTarget target_left;
			target_left.width = width;
			target_left.height = height;
			target_left.framebuffer = (uint64_t*)virt_framebuffer_left->cptr;
			target_left.indexbuffer = nullptr;
			target_left.view = mat4(viewLeft.view); 
			target_left.proj = viewLeft.proj;
			target_left.VP = viewLeft.VP;
			target_left.isLeft = true;
			target_left.isRight = false;

			RenderTarget target_right;
			target_right.width = width;
			target_right.height = height;
			target_right.framebuffer = (uint64_t*)virt_framebuffer_right->cptr;
			target_right.indexbuffer = nullptr;
			target_right.view = mat4(viewRight.view); 
			target_right.proj = viewRight.proj;
			target_right.VP = viewRight.VP;
			target_right.isLeft = false;
			target_right.isRight = true;

			vector<RenderTarget> targets = { target_left, target_right };
			draw(&scene, targets);

			Rectangle targetViewport;
			targetViewport.x = 0;
			targetViewport.y = 0;
			targetViewport.width = width;
			targetViewport.height = height;

			auto glMapping_left = mapCudaGl(viewLeft.framebuffer->colorAttachments[0]);
			auto glMapping_right = mapCudaGl(viewRight.framebuffer->colorAttachments[0]);

			prog_gaussians_rendering->launch("kernel_blit_opengl", 
				{&launchArgs, &target_left, &glMapping_left.surface, &targetViewport}, 
				targetViewport.width * targetViewport.height);
			prog_gaussians_rendering->launch("kernel_blit_opengl", 
				{&launchArgs, &target_right, &glMapping_right.surface, &targetViewport}, 
				targetViewport.width * targetViewport.height);

			glMapping_left.unmap();
			glMapping_right.unmap();

		}

		// blit vr framebuffers to desktop framebuffer
		glBlitNamedFramebuffer(viewLeft.framebuffer->handle, GLRenderer::view.framebuffer->handle, 
			0, 0, width, height,
			0, 0, GLRenderer::view.framebuffer->width / 2, GLRenderer::view.framebuffer->height,
			GL_COLOR_BUFFER_BIT, GL_LINEAR
		);
		
		glBlitNamedFramebuffer(viewRight.framebuffer->handle, GLRenderer::view.framebuffer->handle, 
			0, 0, width, height,
			GLRenderer::view.framebuffer->width / 2, 0, GLRenderer::view.framebuffer->width, GLRenderer::view.framebuffer->height,
			GL_COLOR_BUFFER_BIT, GL_LINEAR
		);

		if(Runtime::measureTimings){
			cuEventRecord(event_render_end, 0);

			cuCtxSynchronize();

			float duration;
			cuEventElapsedTime(&duration, event_render_start, event_render_end);

			Runtime::timings.add("[render - VR]", duration);
		}

		// DEBUG - INSET
		// int insetSize = 64;
		// glBlitNamedFramebuffer(viewRight.framebuffer->handle, GLRenderer::view.framebuffer->handle, 
		// 	Runtime::mousePosition.x - insetSize / 2, Runtime::mousePosition.y - insetSize / 2, 
		// 	Runtime::mousePosition.x + insetSize / 2, Runtime::mousePosition.y + insetSize / 2,
		// 	500, 0, 500 + 1024, 0 + 1024,
		// 	GL_COLOR_BUFFER_BIT, GL_NEAREST
		// );

	}else{

		// RENDER DESKTOP

		if(Runtime::measureTimings){
			cuEventRecord(event_render_start, 0);
		}

		cuStreamSynchronize(0);

		// { // MAIN
		// 	RenderTarget target;
		// 	target.width = GLRenderer::width;
		// 	target.height = GLRenderer::height;
		// 	target.framebuffer = (uint64_t*)virt_framebuffer->cptr;
		// 	target.indexbuffer = nullptr;
		// 	target.view = mat4(GLRenderer::camera->view);
		// 	target.proj = GLRenderer::camera->proj;
			
		// 	draw(&scene, GLRenderer::view, target, mainstream, sidestream);

		// 	prog_gaussians_editing->launch("kernel_computeHoveredObject", {&launchArgs, &target}, 256, mainstream);

		// 	auto glMapping = mapCudaGl(GLRenderer::view.framebuffer->colorAttachments[0]);
		// 	prog_gaussians_rendering->launch("kernel_toOpenGL", {&launchArgs, &target, &glMapping.surface}, GLRenderer::width * GLRenderer::height, mainstream);
		// 	glMapping.unmap();
		// }

		if(settings.enableStereoFramebufferTest && settings.enableOverlapped){ 
			// render to multiple targets concurrently
			
			static shared_ptr<CudaVirtualMemory> virt_framebuffer_secondary = CURuntime::allocVirtual("secondary framebuffer (stereo concurrency test)");
			virt_framebuffer_secondary->commit(virt_framebuffer->comitted);

			RenderTarget target;
			target.width = GLRenderer::width;
			target.height = GLRenderer::height;
			target.framebuffer = (uint64_t*)virt_framebuffer->cptr;
			target.indexbuffer = nullptr;
			target.view = mat4(GLRenderer::camera->view);
			target.proj = GLRenderer::camera->proj;
			target.VP = GLRenderer::camera->VP;

			RenderTarget secondary = target;
			secondary.framebuffer = (uint64_t*)virt_framebuffer_secondary->cptr;
			
			vector<RenderTarget> targets = { target, secondary};
			draw(&scene, targets);

			Rectangle targetViewport_left;
			targetViewport_left.x = 0;
			targetViewport_left.y = 0;
			targetViewport_left.width = GLRenderer::width / 2;
			targetViewport_left.height = GLRenderer::height / 2;

			Rectangle targetViewport_right;
			targetViewport_right.x = GLRenderer::width / 2;
			targetViewport_right.y = 0;
			targetViewport_right.width = GLRenderer::width / 2;
			targetViewport_right.height = GLRenderer::height / 2;

			auto glMapping = mapCudaGl(GLRenderer::view.framebuffer->colorAttachments[0]);

			prog_gaussians_rendering->launch("kernel_blit_opengl", 
				{&launchArgs, &target, &glMapping.surface, &targetViewport_left}, 
				targetViewport_left.width * targetViewport_left.height);

			prog_gaussians_rendering->launch("kernel_blit_opengl", 
				{&launchArgs, &secondary, &glMapping.surface, &targetViewport_right}, 
				targetViewport_right.width * targetViewport_right.height);

			glMapping.unmap();
		}else if(settings.enableStereoFramebufferTest && !settings.enableOverlapped){ 

			{ // LEFT
				RenderTarget target;
				target.width = GLRenderer::width;
				target.height = GLRenderer::height;
				target.framebuffer = (uint64_t*)virt_framebuffer->cptr;
				target.indexbuffer = nullptr;
				target.view = mat4(GLRenderer::camera->view);
				target.proj = GLRenderer::camera->proj;
				target.VP = GLRenderer::camera->VP;

				vector<RenderTarget> targets = {target};
				draw(&scene, targets);

				Rectangle viewport;
				viewport.x = 0;
				viewport.y = 500;
				viewport.width = GLRenderer::width / 2;
				viewport.height = GLRenderer::height / 2;

				auto glMapping = mapCudaGl(GLRenderer::view.framebuffer->colorAttachments[0]);

				prog_gaussians_rendering->launch("kernel_blit_opengl", 
					{&launchArgs, &target, &glMapping.surface, &viewport}, 
					viewport.width * viewport.height);

				glMapping.unmap();
			}

			{ // RIGHT
				RenderTarget target;
				target.width = GLRenderer::width;
				target.height = GLRenderer::height;
				target.framebuffer = (uint64_t*)virt_framebuffer->cptr;
				target.indexbuffer = nullptr;
				target.view = mat4(GLRenderer::camera->view);
				target.proj = GLRenderer::camera->proj;
				target.VP = GLRenderer::camera->VP;

				vector<RenderTarget> targets = {target};
				draw(&scene, targets);

				Rectangle viewport;
				viewport.x = GLRenderer::width / 2;
				viewport.y = 500;
				viewport.width = GLRenderer::width / 2;
				viewport.height = GLRenderer::height / 2;

				auto glMapping = mapCudaGl(GLRenderer::view.framebuffer->colorAttachments[0]);

				prog_gaussians_rendering->launch("kernel_blit_opengl", 
					{&launchArgs, &target, &glMapping.surface, &viewport}, 
					viewport.width * viewport.height);

				glMapping.unmap();
			}



		}else{
			// standard single-target-rendering
			RenderTarget target;
			target.width = GLRenderer::width;
			target.height = GLRenderer::height;
			target.framebuffer = (uint64_t*)virt_framebuffer->cptr;
			target.indexbuffer = nullptr;
			target.view = mat4(GLRenderer::camera->view);
			target.proj = GLRenderer::camera->proj;
			target.VP = GLRenderer::camera->VP;
			
			vector<RenderTarget> targets = { target };
			draw(&scene, targets);
			target = targets[0];

			prog_gaussians_editing->launch("kernel_computeHoveredObject", {&launchArgs, &target}, 256, mainstream);

			auto glMapping = mapCudaGl(GLRenderer::view.framebuffer->colorAttachments[0]);
			prog_gaussians_rendering->launch("kernel_toOpenGL", {&launchArgs, &target, &glMapping.surface}, GLRenderer::width * GLRenderer::height, mainstream);
			glMapping.unmap();
		}




		if(Runtime::measureTimings){
			cuEventRecord(event_render_end, 0);

			cuCtxSynchronize();

			float duration;
			cuEventElapsedTime(&duration, event_render_start, event_render_end);

			Runtime::timings.add("[render - desktop]", duration);
		}

		if(settings.showInset)
		{ // BLIT AN INSET
			cuCtxSynchronize();
			
			int insetSize = 32;
			int factor = 16;
			ivec2 center = {16 * 60 + 8, 16 * 50 + 8};
			// int start = {16 * 60 - 8, 16 * 50 - 8};
			// int end =   {16 * 60 + 8, 16 * 50 + 8};

			// first copy source area to bottom-left
			// ivec2 center = ivec2{GLRenderer::width / 2, GLRenderer::height / 2};
			glBlitNamedFramebuffer(GLRenderer::view.framebuffer->handle, GLRenderer::view.framebuffer->handle, 
				center.x - insetSize / 2, center.y - insetSize / 2, center.x + insetSize / 2, center.y + insetSize / 2,
				0, 0, insetSize, insetSize,
				GL_COLOR_BUFFER_BIT, GL_NEAREST
			);

			// then copy and rescale to target area
			glBlitNamedFramebuffer(GLRenderer::view.framebuffer->handle, GLRenderer::view.framebuffer->handle, 
				0, 0, insetSize, insetSize,
				center.x - factor * insetSize / 2, 
				10, //center.y - factor * insetSize / 2, 
				center.x + factor * insetSize / 2, 
				10 + factor * insetSize, // center.y + factor * insetSize / 2,
				GL_COLOR_BUFFER_BIT, GL_NEAREST
			);

			// make border in source area
			// left
			glBlitNamedFramebuffer(GLRenderer::view.framebuffer->handle, GLRenderer::view.framebuffer->handle, 
				200, 200, 1, 1,
				center.x - insetSize / 2, center.y - insetSize / 2, center.x - insetSize / 2 + 1, center.y + insetSize / 2,
				GL_COLOR_BUFFER_BIT, GL_NEAREST
			);
			// right
			glBlitNamedFramebuffer(GLRenderer::view.framebuffer->handle, GLRenderer::view.framebuffer->handle, 
				200, 200, 1, 1,
				center.x + insetSize / 2 - 1, center.y - insetSize / 2, center.x + insetSize / 2, center.y + insetSize / 2,
				GL_COLOR_BUFFER_BIT, GL_NEAREST
			);
			// bottom
			glBlitNamedFramebuffer(GLRenderer::view.framebuffer->handle, GLRenderer::view.framebuffer->handle, 
				200, 200, 1, 1,
				center.x - insetSize / 2, center.y - insetSize / 2, center.x + insetSize / 2, center.y - insetSize / 2 + 1,
				GL_COLOR_BUFFER_BIT, GL_NEAREST
			);
			// top
			glBlitNamedFramebuffer(GLRenderer::view.framebuffer->handle, GLRenderer::view.framebuffer->handle, 
				200, 200, 1, 1,
				center.x - insetSize / 2, center.y + insetSize / 2 - 1, center.x + insetSize / 2, center.y + insetSize / 2,
				GL_COLOR_BUFFER_BIT, GL_NEAREST
			);
		}
	}

	cuCtxSynchronize();
	CudaModularProgram::resolveTimings();
	
	// DRAW GUI
	if(!ovr->isActive()){

		ImGui::SetCurrentContext(imguicontext_desktop);

		drawGUI();

		// { // show vr gui in desktop mode
		// 	imguiStyleVR();
		// 	makePaintingVR(imn_painting->page);
		// 	ImGui::StyleColorsDark();
		// }

		Runtime::totalTileFragmentCount = 0;

		ImGui::Render();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
	}else{

		{ // render desktop gui while in VR
			ImGui::SetCurrentContext(imguicontext_desktop);
			ImGuiIO& io = ImGui::GetIO();

			auto fbo = GLRenderer::view.framebuffer;
			glBindFramebuffer(GL_FRAMEBUFFER, fbo->handle);
			glViewport(0, 0, fbo->width, fbo->height);

			drawGUI();
			ImGui::Render();
			ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
		}

		Runtime::totalTileFragmentCount = 0;
	}

	cuStreamSynchronize(0);
	cuStreamSynchronize(mainstream);
	cuCtxSynchronize();
	
	mouse_prev = Runtime::mouseEvents;

	if(viewmode == VIEWMODE_IMMERSIVE_VR && ovr->isActive()){

		auto size = ovr->getRecommmendedRenderTargetSize();	
		int width = size[0];
		int height = size[1];	

		vr::VRTextureBounds_t bounds = {
			0.0f, 
			1.0f - float(height) / float(viewLeft.framebuffer->height),
			float(width) / float(viewLeft.framebuffer->width), 
			1.0f,
		};

		ovr->submit(viewLeft.framebuffer->colorAttachments[0]->handle, vr::EVREye::Eye_Left, bounds);
		ovr->submit(viewRight.framebuffer->colorAttachments[0]->handle, vr::EVREye::Eye_Right, bounds);

		ovr->postPresentHandoff();
	}

	cuMemcpyDtoHAsync(h_state_pinned, cptr_state, sizeof(DeviceState), mainstream);
	cuStreamSynchronize(mainstream);
	memcpy(&deviceState, h_state_pinned, sizeof(DeviceState));

	Runtime::numSelectedSplats = deviceState.numSelectedSplats;

	settings.shortcutsDisabledForXFrames = max(settings.shortcutsDisabledForXFrames - 1, 0);

	Runtime::mouseEvents.clear();

	if(Runtime::measureTimings){
		CudaModularProgram::clearTimings();
	}

	ImGui::SetCurrentContext(imguicontext_desktop);

	postRenderStuff();
}
