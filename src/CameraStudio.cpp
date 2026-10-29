#include "CameraStudio.h"
#include "CameraPath.h"
#include "SplatEditor.h"
#include "writer/MjpegAviWriter.h"
#include "imgui_internal.h"
#include "json/json.hpp"

#include <array>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <format>
#include <limits>
#include <string>

namespace {
using Json = nlohmann::json;
namespace fs = std::filesystem;

struct DisabledScope {
	bool disabled;
	explicit DisabledScope(bool value) : disabled(value) {
		ImGui::PushItemFlag(ImGuiItemFlags_Disabled, value);
		if(value) ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.45f);
	}
	~DisabledScope() {
		if(disabled) ImGui::PopStyleVar();
		ImGui::PopItemFlag();
	}
};

bool finite(glm::vec3 value) {
	return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

CameraKeyframe currentCamera() {
	CameraKeyframe key;
	key.position = glm::vec3(GLRenderer::camera->world[3]);
	key.orientation = cameraNormalizeQuaternion(glm::quat_cast(glm::mat3(GLRenderer::camera->world)));
	key.fovy = float(GLRenderer::camera->fovy);
	return key;
}

void applyMainCamera(const CameraKeyframe& key) {
	auto& camera = *GLRenderer::camera;
	camera.world = key.world(); camera.position = key.position; camera.fovy = key.fovy;
	camera.update();
}

bool projectMain(glm::vec3 world, ImVec2& screen) {
	auto clip = glm::mat4(GLRenderer::camera->proj * GLRenderer::camera->view) * glm::vec4(world, 1);
	if(clip.w < 0.0001f) return false;
	screen = {(clip.x / clip.w * 0.5f + 0.5f) * GLRenderer::width,
		(0.5f - clip.y / clip.w * 0.5f) * GLRenderer::height};
	return std::isfinite(screen.x) && std::isfinite(screen.y);
}

float distanceSquared(ImVec2 a, ImVec2 b) {
	return (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y);
}
}

struct CameraStudio::Impl {
	CameraPath path;
	std::vector<CameraPath> undoStack, redoStack;
	int selected = -1;
	bool editorOpen = false, previewOpen = false;
	bool playing = false, timelinePreview = false, loopPlayback = false, followMain = false;
	bool editingMain = false, overviewSaved = false, navGesture = false, gizmoGesture = false;
	bool viewportCaptured = false, previewVisible = false, previewDragging = false;
	float playhead = 0, playbackSpeed = 1, moveSpeed = 1;
	OrbitControls overviewControls;
	double overviewFovy = 47.136211;
	ImGuizmo::OPERATION operation = ImGuizmo::TRANSLATE;
	bool localGizmo = false, showFrustums = true;
	float markerSize = 0.1f;

	int plane = 0, mapMode = 0, mapDragKey = -1, aimMode = 0, generatorCount = 12;
	glm::vec3 mapCenter{0}, aimTarget{0};
	float mapSpan = 10, planeOffset = 0, orbitRadius = 3, orbitHeight = 1, orbitArc = 360;
	bool appendGenerated = false, drawing = false;
	std::vector<glm::vec3> stroke, mapPoints;

	int outputWidth = 1280, outputHeight = 720, outputFps = 30, jpegQuality = 90;
	bool overwriteVideo = false, recording = false;
	uint32_t totalFrames = 0;
	int recordingWidth = 0, recordingHeight = 0, recordingQuality = 90;
	MjpegAviWriter video;
	char pathFile[1024] = "camera_path.json";
	char videoFile[1024] = "camera_path.avi";
	std::string status;
	std::shared_ptr<Framebuffer> previewFramebuffer;
	std::shared_ptr<CudaVirtualMemory> previewPixels;
	CUdeviceptr previewState = 0, emptyLineCount = 0;

	bool validSelection() const { return selected >= 0 && selected < int(path.keys.size()); }
	bool active() const { return editorOpen || previewOpen || recording; }
	void checkpoint() {
		undoStack.push_back(path);
		if(undoStack.size() > 100) undoStack.erase(undoStack.begin());
		redoStack.clear();
	}
	void select(int index) { selected = index; timelinePreview = false; playing = false; }
	void changed() { playing = timelinePreview = false; playhead = std::min(playhead, path.duration()); }
	void itemCheckpoint() { if(ImGui::IsItemActivated()) checkpoint(); }
	void history(bool redo) {
		auto& from = redo ? redoStack : undoStack;
		auto& to = redo ? undoStack : redoStack;
		if(from.empty()) return;
		to.push_back(path); path = std::move(from.back()); from.pop_back();
		selected = path.keys.empty() ? -1 : std::clamp(selected, 0, int(path.keys.size()) - 1);
		playing = false; changed();
	}
	void saveOverview() {
		if(overviewSaved) return;
		overviewControls = *Runtime::controls;
		overviewFovy = GLRenderer::camera->fovy;
		overviewSaved = true;
	}
	void restoreOverview() {
		if(!overviewSaved) return;
		*Runtime::controls = overviewControls;
		Runtime::controls->isLeftDown = Runtime::controls->isRightDown = false;
		Runtime::controls->mousePos = {Runtime::mouseEvents.pos_x, Runtime::mouseEvents.pos_y};
		Runtime::controls->update(); GLRenderer::camera->world = Runtime::controls->world;
		GLRenderer::camera->position = glm::dvec3(GLRenderer::camera->world[3]);
		GLRenderer::camera->fovy = overviewFovy; GLRenderer::camera->update();
		overviewSaved = false;
	}
	void consumeMouse() {
		Runtime::controls->isLeftDown = Runtime::controls->isRightDown = Runtime::controls->isMiddleDown = false;
		Runtime::controls->mousePos = {Runtime::mouseEvents.pos_x, Runtime::mouseEvents.pos_y};
	}
	void addCurrent(bool insert) {
		if(path.keys.size() >= 512) { status = "Camera limit: 512."; return; }
		checkpoint(); auto key = currentCamera();
		int index = insert && validSelection() ? selected + 1 : int(path.keys.size());
		path.keys.insert(path.keys.begin() + index, key); select(index); changed();
	}
	void refreshMap(SplatEditor& editor);
	void generate(std::vector<glm::vec3> positions, bool closed);
	void drawMap();
	void drawOverlay();
	void drawEditor(SplatEditor& editor);
	void drawPreview(SplatEditor& editor);
	void renderPose(SplatEditor& editor, const CameraKeyframe& pose, int width, int height);
	void savePath();
	void loadPath();
	void startRecording(SplatEditor& editor);
	void stopRecording(bool completed);
	CameraKeyframe previewPose() const {
		if(recording || timelinePreview || playing) return path.sample(playhead);
		return validSelection() ? path.keys[selected] : currentCamera();
	}
};

CameraStudio::CameraStudio() : impl(std::make_unique<Impl>()) {}
CameraStudio::~CameraStudio() = default;
bool CameraStudio::isOpen() const { return impl->active(); }

void CameraStudio::close() {
	if(impl->recording) impl->stopRecording(false);
	impl->editorOpen = impl->previewOpen = impl->playing = impl->editingMain = false;
	impl->restoreOverview();
}

void CameraStudio::open(SplatEditor& editor) {
	auto& s = *impl;
	if(editor.ovr->isActive()) { s.status = "Camera Studio is available in desktop mode."; return; }
	editor.setAction(nullptr);
	editor.settings.multiView = false;
	editor.settings.enableStereoFramebufferTest = false;
	s.editorOpen = s.previewOpen = true;
	if(s.path.keys.empty()) {
		s.mapCenter = s.aimTarget = glm::vec3(Runtime::controls->target);
		s.mapSpan = std::max(float(Runtime::controls->radius) * 2.5f, 0.1f);
		s.orbitRadius = std::max(float(Runtime::controls->radius), 0.01f);
		s.orbitHeight = currentCamera().position.z - s.aimTarget.z;
		s.planeOffset = currentCamera().position.z;
		s.markerSize = s.orbitRadius * 0.03f;
		s.moveSpeed = std::max(s.orbitRadius * 0.3f, 0.01f);
	}
	s.refreshMap(editor);
}

void CameraStudio::Impl::refreshMap(SplatEditor& editor) {
	// 按需抽样场景点作为绘图底图，不增加常驻 CUDA 内核。
	mapPoints.clear(); std::vector<SNSplats*> nodes;
	editor.scene.process<SNSplats>([&](SNSplats* node) {
		if(node->visible && !node->hidden && node->dmng.data.count) nodes.push_back(node);
	});
	if(nodes.empty()) return;
	cuCtxSynchronize();
	for(auto node : nodes) {
		auto& data = node->dmng.data;
		size_t count = std::min<size_t>(data.count, std::max<size_t>(1, 12000 / nodes.size()));
		size_t stride = std::max<size_t>(1, size_t(data.count) / count);
		std::vector<glm::vec3> positions(count); std::vector<uint32_t> flags(count);
		CUDA_MEMCPY2D copy{};
		copy.srcMemoryType = CU_MEMORYTYPE_DEVICE; copy.dstMemoryType = CU_MEMORYTYPE_HOST;
		copy.srcDevice = (CUdeviceptr)data.position; copy.srcPitch = stride * sizeof(glm::vec3);
		copy.dstHost = positions.data(); copy.dstPitch = sizeof(glm::vec3);
		copy.WidthInBytes = sizeof(glm::vec3); copy.Height = count;
		if(cuMemcpy2D(&copy) != CUDA_SUCCESS) continue;
		copy.srcDevice = (CUdeviceptr)data.flags; copy.srcPitch = stride * sizeof(uint32_t);
		copy.dstHost = flags.data(); copy.dstPitch = sizeof(uint32_t); copy.WidthInBytes = sizeof(uint32_t);
		if(cuMemcpy2D(&copy) != CUDA_SUCCESS) continue;
		for(size_t i = 0; i < count; ++i) {
			auto point = glm::vec3(node->transform_global * glm::vec4(positions[i], 1));
			if(!(flags[i] & FLAGS_DELETED) && finite(point)) mapPoints.push_back(point);
		}
	}
	status = "Drawing map refreshed.";
}

void CameraStudio::Impl::generate(std::vector<glm::vec3> positions, bool closed) {
	if(positions.size() < 2) { status = "Draw a longer stroke to generate a path."; return; }
	if(positions.size() + (appendGenerated ? path.keys.size() : 0) > 512) {
		status = "Camera limit: 512."; return;
	}
	checkpoint();
	if(!appendGenerated) path.keys.clear();
	auto base = currentCamera();
	for(size_t i = 0; i < positions.size(); ++i) {
		auto key = base; key.position = positions[i];
		if(aimMode == 0) key.aimAt(aimTarget);
		if(aimMode == 1) {
			auto next = i + 1 < positions.size() ? positions[i + 1] :
				(closed ? positions.front() : positions[i] + positions[i] - positions[i - 1]);
			key.aimAt(next);
		}
		path.keys.push_back(key);
	}
	path.closed = closed && !appendGenerated;
	select(0); changed(); status = "Path generated. Select a camera to adjust its pose.";
}

bool CameraStudio::handleDesktopInput(SplatEditor& editor) {
	auto& s = *impl;
	if(!s.active() || editor.settings.hideGUI || editor.settings.multiView ||
		editor.settings.enableStereoFramebufferTest || editor.ovr->isActive()) {
		if(s.recording) s.stopRecording(false);
		s.playing = s.editingMain = false; s.restoreOverview(); return false;
	}
	if(ImGui::GetIO().WantCaptureKeyboard) editor.temporarilyDisableShortcuts();
	if(s.playing && !s.recording) {
		s.playhead += float(GLRenderer::timeSinceLastFrame) * s.playbackSpeed;
		float duration = s.path.duration();
		if(s.playhead >= duration) {
			if(s.loopPlayback && duration > 0) s.playhead = std::fmod(s.playhead, duration);
			else { s.playhead = duration; s.playing = false; }
		}
	}
	if(s.recording) {
		// 录制只按帧号推进，与机器实时帧率无关。
		s.playhead = s.path.duration() * float(s.video.frameCount()) /
			float(s.path.closed ? s.totalFrames : s.totalFrames - 1);
	}
	if(s.followMain && (s.timelinePreview || s.playing || s.recording)) {
		s.saveOverview(); applyMainCamera(s.previewPose()); s.consumeMouse(); return true;
	}
	if(s.editingMain && s.validSelection() && !s.recording) {
		auto& key = s.path.keys[s.selected]; auto& io = ImGui::GetIO();
		auto delta = glm::vec2(io.MouseDelta.x, -io.MouseDelta.y);
		bool mouse = !io.WantCaptureMouse;
		bool keyboard = !io.WantCaptureKeyboard && !io.WantTextInput;
		bool rotate = mouse && io.MouseDown[0] && glm::length(delta) > 0;
		bool pan = mouse && io.MouseDown[1] && glm::length(delta) > 0;
		float wheel = mouse ? io.MouseWheel : 0;
		glm::vec3 movement(0);
		if(keyboard) {
			movement.x = float(glfwGetKey(GLRenderer::window, GLFW_KEY_D) == GLFW_PRESS) - float(glfwGetKey(GLRenderer::window, GLFW_KEY_A) == GLFW_PRESS);
			movement.y = float(glfwGetKey(GLRenderer::window, GLFW_KEY_E) == GLFW_PRESS) - float(glfwGetKey(GLRenderer::window, GLFW_KEY_Q) == GLFW_PRESS);
			movement.z = float(glfwGetKey(GLRenderer::window, GLFW_KEY_S) == GLFW_PRESS) - float(glfwGetKey(GLRenderer::window, GLFW_KEY_W) == GLFW_PRESS);
		}
		bool navigation = rotate || pan || wheel != 0 || glm::length(movement) > 0;
		if(navigation && !s.navGesture) s.checkpoint();
		s.navGesture = navigation;
		if(rotate) {
			auto yaw = cameraAngleAxis(-delta.x * 0.003f, glm::vec3(0, 0, 1));
			auto pitch = cameraAngleAxis(delta.y * 0.003f, glm::vec3(1, 0, 0));
			key.orientation = cameraNormalizeQuaternion(yaw * key.orientation * pitch);
		}
		auto rotation = glm::mat3_cast(key.orientation);
		if(pan) key.position += rotation * glm::vec3(-delta.x, -delta.y, 0) * (s.moveSpeed * 0.002f);
		float boost = io.KeyShift ? 4.0f : (io.KeyCtrl ? 0.25f : 1.0f);
		key.position += rotation * movement * s.moveSpeed * boost * float(std::min(GLRenderer::timeSinceLastFrame, 0.1));
		key.position -= rotation[2] * wheel * s.moveSpeed * 0.15f;
		applyMainCamera(key); s.consumeMouse(); return true;
	}
	s.restoreOverview();
	if(s.recording) { s.consumeMouse(); return true; }
	// gizmo 拖动也会捕获鼠标，不能因此跳过下一帧的 Manipulate。
	if(ImGui::IsWindowHovered(ImGuiHoveredFlags_AnyWindow) && !s.gizmoGesture) {
		s.consumeMouse(); return true;
	}
	if(editor.state.currentAction || editor.settings.enableStereoFramebufferTest) return false;
	if(s.validSelection() && !s.playing && !ImGui::GetIO().WantCaptureKeyboard) {
		if(Runtime::getKeyAction('t') == GLFW_PRESS) { s.operation = ImGuizmo::TRANSLATE; return true; }
		if(Runtime::getKeyAction('r') == GLFW_PRESS) { s.operation = ImGuizmo::ROTATE; return true; }
		// 场景中的删除键只删除当前相机，不影响模型。
		if(Runtime::getKeyAction(GLFW_KEY_DELETE) == GLFW_PRESS) {
			s.checkpoint(); s.path.keys.erase(s.path.keys.begin() + s.selected);
			s.selected = s.path.keys.empty() ? -1 : std::min(s.selected, int(s.path.keys.size()) - 1);
			s.changed(); return true;
		}
	}
	if(s.validSelection() && !s.playing) {
		glm::mat4 world = s.path.keys[s.selected].world();
		glm::mat4 view = GLRenderer::camera->view, projection = GLRenderer::camera->proj;
		ImGuizmo::SetID(9351); ImGuizmo::SetOrthographic(false);
		ImGuizmo::SetDrawlist(ImGui::GetBackgroundDrawList());
		ImGuizmo::SetRect(0, 0, float(GLRenderer::width), float(GLRenderer::height));
		bool modified = ImGuizmo::Manipulate(&view[0][0], &projection[0][0], s.operation,
			s.localGizmo ? ImGuizmo::LOCAL : ImGuizmo::WORLD, &world[0][0]);
		bool usingGizmo = ImGuizmo::IsUsing();
		if(usingGizmo && !s.gizmoGesture) s.checkpoint();
		s.gizmoGesture = usingGizmo;
		if(modified) {
			s.path.keys[s.selected].position = glm::vec3(world[3]);
			s.path.keys[s.selected].orientation = cameraNormalizeQuaternion(glm::quat_cast(glm::mat3(world)));
			s.changed();
		}
		if(usingGizmo || (ImGuizmo::IsOver() && ImGui::IsMouseDown(0))) {
			s.consumeMouse(); return true;
		}
	}
	if(ImGui::GetIO().WantCaptureMouse) { s.consumeMouse(); return true; }
	if(s.viewportCaptured) {
		if(!ImGui::IsMouseDown(0)) s.viewportCaptured = false;
		s.consumeMouse(); return true;
	}
	if(!ImGui::GetIO().WantCaptureMouse && ImGui::IsMouseClicked(0) && s.showFrustums) {
		int closest = -1; float distance = 16 * 16;
		for(int i = 0; i < int(s.path.keys.size()); ++i) {
			ImVec2 screen;
			if(projectMain(s.path.keys[i].position, screen)) {
				float d = distanceSquared(screen, ImGui::GetMousePos());
				if(d < distance) { distance = d; closest = i; }
			}
		}
		if(closest >= 0) {
			s.select(closest); s.previewOpen = true; s.viewportCaptured = true;
			s.consumeMouse(); return true;
		}
	}
	return false;
}

void CameraStudio::Impl::renderPose(SplatEditor& editor, const CameraKeyframe& pose, int width, int height) {
	if(!previewFramebuffer) {
		previewFramebuffer = GLRenderer::createFramebuffer(width, height);
		previewPixels = CURuntime::allocVirtual("camera studio pixels");
		previewState = CURuntime::alloc("camera studio state", sizeof(DeviceState));
		emptyLineCount = CURuntime::alloc("camera studio empty lines", 8);
		cuMemsetD8(emptyLineCount, 0, 8);
	}
	previewFramebuffer->setSize(width, height);
	previewPixels->commit(uint64_t(width) * height * 8);
	// 独立状态和空线列表隔离预览，保留主视图的拾取、测量及编辑数据。
	struct Restore {
		SplatEditor& editor;
		CommonLaunchArgs args;
		CUdeviceptr lines, hostLines;
		bool soa, bandwidth, intersections;
		std::vector<TriangleQueueItem> triangles;
		int64_t splats, fragments, renderedTriangles;
		int tileFragments;
		std::vector<std::pair<std::string, std::string>> debugValues;
#ifdef SPLATSHOP_ENABLE_GAUSSIAN_SELECTOR
		std::shared_ptr<GaussianSelectorRuntime> selector;
#endif
		explicit Restore(SplatEditor& e) : editor(e), args(e.launchArgs), lines(e.cptr_numLines),
			hostLines(e.cptr_numLines_host), soa(e.settings.renderSoA), bandwidth(e.settings.renderBandwidth),
			intersections(e.settings.renderFragIntersections), triangles(std::move(e.triangleQueue)),
			splats(Runtime::numVisibleSplats), fragments(Runtime::numVisibleFragments),
			renderedTriangles(Runtime::numRenderedTriangles), tileFragments(Runtime::totalTileFragmentCount),
			debugValues(Runtime::debugValueList)
#ifdef SPLATSHOP_ENABLE_GAUSSIAN_SELECTOR
			, selector(std::move(e.gaussianSelector))
#endif
		{}
		~Restore() {
			editor.launchArgs = args; editor.cptr_numLines = lines; editor.cptr_numLines_host = hostLines;
			editor.settings.renderSoA = soa; editor.settings.renderBandwidth = bandwidth;
			editor.settings.renderFragIntersections = intersections;
			editor.triangleQueue = std::move(triangles);
			Runtime::numVisibleSplats = splats; Runtime::numVisibleFragments = fragments;
			Runtime::numRenderedTriangles = renderedTriangles; Runtime::totalTileFragmentCount = tileFragments;
			Runtime::debugValueList = std::move(debugValues);
#ifdef SPLATSHOP_ENABLE_GAUSSIAN_SELECTOR
			editor.gaussianSelector = std::move(selector);
#endif
		}
	} restore(editor);
	editor.triangleQueue.clear();
	// 复用支持独立相机颜色的标准路径；实验布局只影响主视图。
	editor.settings.renderSoA = editor.settings.renderBandwidth = editor.settings.renderFragIntersections = false;
	cuCtxSynchronize(); cuMemsetD8(previewState, 0, sizeof(DeviceState));
	editor.launchArgs.state = (DeviceState*)previewState;
	editor.launchArgs.brush = {}; editor.launchArgs.rectselect = {};
	editor.launchArgs.mouseEvents = {}; editor.launchArgs.mouseEvents.pos_x = -1; editor.launchArgs.mouseEvents.pos_y = -1;
	editor.launchArgs.mouseEvents_prev = editor.launchArgs.mouseEvents;
	editor.launchArgs.keyEvents.numEvents = 0;
	editor.launchArgs.uniforms.camWorld = pose.world();
	editor.launchArgs.uniforms.inset.show = false;
	editor.launchArgs.uniforms.measurementDepthMode = 0;
	editor.cptr_numLines = editor.cptr_numLines_host = emptyLineCount;
	RenderTarget target{};
	target.cameraPreview = true;
	target.width = width; target.height = height;
	target.framebuffer = (uint64_t*)previewPixels->cptr;
	target.view = glm::inverse(pose.world());
	float nearPlane = float(GLRenderer::camera->near), farPlane = float(GLRenderer::camera->far);
	float aspect = float(width) / height, fov = glm::radians(pose.fovy);
	target.proj = Camera::createProjectionMatrix(nearPlane, fov, aspect);
	target.VP = Camera::createVP(nearPlane, farPlane, fov, aspect, width, height);
	std::vector<RenderTarget> targets{target};
	editor.draw(&editor.scene, targets); target = targets[0];
	auto mapping = editor.mapCudaGl(previewFramebuffer->colorAttachments[0]);
	editor.prog_gaussians_rendering->launch("kernel_toOpenGL",
		{&editor.launchArgs, &target, &mapping.surface}, width * height, editor.mainstream);
	cuStreamSynchronize(editor.mainstream); mapping.unmap();
}

void CameraStudio::render(SplatEditor& editor) {
	auto& s = *impl;
	if(!s.active() || editor.ovr->isActive() || editor.settings.hideGUI || editor.settings.multiView ||
		editor.settings.enableStereoFramebufferTest) {
		if(s.recording) s.stopRecording(false);
		return;
	}
	if(!s.recording && (!s.previewOpen || !s.previewVisible)) return;
	int width = s.recording ? s.recordingWidth : std::min(s.outputWidth, 960);
	int height = s.recording ? s.recordingHeight : std::max(16, int(float(width) * s.outputHeight / s.outputWidth));
	try {
		s.renderPose(editor, s.previewPose(), width, height);
		if(s.recording) {
			std::vector<uint8_t> rgba(size_t(width) * height * 4), rgb(size_t(width) * height * 3);
			GLint packAlignment, packBuffer;
			glGetIntegerv(GL_PACK_ALIGNMENT, &packAlignment); glGetIntegerv(GL_PIXEL_PACK_BUFFER_BINDING, &packBuffer);
			glBindBuffer(GL_PIXEL_PACK_BUFFER, 0); glPixelStorei(GL_PACK_ALIGNMENT, 1);
			glGetTextureImage(s.previewFramebuffer->colorAttachments[0]->handle, 0, GL_RGBA, GL_UNSIGNED_BYTE,
				GLsizei(rgba.size()), rgba.data());
			glPixelStorei(GL_PACK_ALIGNMENT, packAlignment); glBindBuffer(GL_PIXEL_PACK_BUFFER, GLuint(packBuffer));
			// OpenGL 从下往上存储，视频帧转换为从上往下的 RGB。
			for(int y = 0; y < height; ++y) for(int x = 0; x < width; ++x) {
				size_t source = (size_t(height - 1 - y) * width + x) * 4;
				size_t destination = (size_t(y) * width + x) * 3;
				std::copy_n(rgba.data() + source, 3, rgb.data() + destination);
			}
			s.video.append(rgb, s.recordingQuality);
			if(s.video.frameCount() == s.totalFrames) s.stopRecording(true);
		}
	} catch(const std::exception& error) {
		std::string message = error.what();
		if(s.recording) s.stopRecording(false);
		s.status = message;
	}
	// 离屏渲染不改变后续 ImGui 的绘制目标。
	glBindFramebuffer(GL_FRAMEBUFFER, GLRenderer::view.framebuffer->handle);
	glViewport(0, 0, GLRenderer::width, GLRenderer::height);
}

void CameraStudio::Impl::drawMap() {
	int axisX = plane == 2 ? 1 : 0, axisY = plane == 0 ? 1 : 2;
	int axisDepth = plane == 0 ? 2 : (plane == 1 ? 1 : 0);
	ImVec2 origin = ImGui::GetCursorScreenPos();
	ImVec2 size(std::max(ImGui::GetContentRegionAvail().x, 80.0f), 235);
	ImGui::InvisibleButton("Path drawing canvas", size);
	bool hovered = ImGui::IsItemHovered(); auto& io = ImGui::GetIO();
	float scale = size.x / mapSpan;
	auto screen = [&](glm::vec3 point) {
		return ImVec2(origin.x + size.x * 0.5f + (point[axisX] - mapCenter[axisX]) * scale,
			origin.y + size.y * 0.5f - (point[axisY] - mapCenter[axisY]) * scale);
	};
	auto world = [&](ImVec2 point) {
		auto value = mapCenter;
		value[axisX] += (point.x - origin.x - size.x * 0.5f) / scale;
		value[axisY] -= (point.y - origin.y - size.y * 0.5f) / scale;
		value[axisDepth] = planeOffset; return value;
	};
	auto draw = ImGui::GetWindowDrawList();
	draw->AddRectFilled(origin, {origin.x + size.x, origin.y + size.y}, IM_COL32(17, 22, 30, 255));
	draw->PushClipRect(origin, {origin.x + size.x, origin.y + size.y}, true);
	float gridStep = std::pow(10.0f, std::floor(std::log10(mapSpan / 8)));
	for(int axis : {axisX, axisY}) {
		float range = mapSpan * (axis == axisX ? 0.5f : size.y / size.x * 0.5f);
		float first = std::floor((mapCenter[axis] - range) / gridStep) * gridStep;
		for(float value = first; value < mapCenter[axis] + range && value < first + gridStep * 200; value += gridStep) {
			auto a = mapCenter, b = mapCenter;
			a[axis] = b[axis] = value;
			int other = axis == axisX ? axisY : axisX;
			a[other] -= mapSpan; b[other] += mapSpan;
			draw->AddLine(screen(a), screen(b), IM_COL32(55, 65, 80, 150));
		}
	}
	for(auto point : mapPoints) draw->AddCircleFilled(screen(point), 1, IM_COL32(120, 139, 155, 65));
	ImVec2 target = screen(aimTarget);
	draw->AddLine({target.x - 6, target.y}, {target.x + 6, target.y}, IM_COL32(115, 200, 255, 255));
	draw->AddLine({target.x, target.y - 6}, {target.x, target.y + 6}, IM_COL32(115, 200, 255, 255));
	if(path.segmentCount() > 0) {
		int steps = std::min(1024, path.segmentCount() * 32);
		for(int i = 1; i <= steps; ++i)
			draw->AddLine(screen(path.sample(path.duration() * (i - 1) / steps).position),
				screen(path.sample(path.duration() * i / steps).position), IM_COL32(150, 187, 255, 255), 2);
	}
	int hoveredKey = -1; float bestDistance = 12 * 12;
	for(int i = 0; i < int(path.keys.size()); ++i) {
		auto keyPosition = screen(path.keys[i].position);
		float distance = distanceSquared(keyPosition, io.MousePos);
		if(distance < bestDistance) { bestDistance = distance; hoveredKey = i; }
		ImU32 color = i == selected ? IM_COL32(255, 200, 70, 255) : IM_COL32(155, 190, 255, 255);
		draw->AddCircleFilled(keyPosition, i == selected ? 6 : 4, color);
		auto forward = path.keys[i].orientation * glm::vec3(0, 0, -1);
		auto direction = screen(path.keys[i].position + forward * mapSpan * 0.05f);
		draw->AddLine(keyPosition, direction, color, 1.5f);
		std::string label = std::to_string(i + 1);
		draw->AddText({keyPosition.x + 7, keyPosition.y - 7}, color, label.c_str());
	}
	for(size_t i = 1; i < stroke.size(); ++i)
		draw->AddLine(screen(stroke[i - 1]), screen(stroke[i]), IM_COL32(255, 190, 70, 255), 2);
	if(timelinePreview || recording)
		draw->AddCircle(screen(path.sample(playhead).position), 9, IM_COL32(100, 245, 170, 255), 16, 2);
	draw->PopClipRect();
	if(recording || editingMain) return;
	if(hovered && ImGui::IsMouseClicked(0)) {
		if(hoveredKey >= 0) {
			select(hoveredKey); checkpoint(); mapDragKey = hoveredKey; previewOpen = true;
		} else if(mapMode == 1 && path.keys.size() < 512) {
			checkpoint(); auto key = currentCamera(); key.position = world(io.MousePos);
			if(aimMode == 0) key.aimAt(aimTarget);
			path.keys.push_back(key); select(int(path.keys.size()) - 1); changed();
		} else if(mapMode == 2) {
			drawing = true; stroke = {world(io.MousePos)};
		}
	}
	if(mapDragKey >= 0 && mapDragKey < int(path.keys.size()) && ImGui::IsMouseDragging(0)) {
		auto point = world(io.MousePos);
		// 在二维平面上拖动时，保留相机原有的第三轴坐标和朝向。
		point[axisDepth] = path.keys[mapDragKey].position[axisDepth];
		path.keys[mapDragKey].position = point; changed();
	}
	if(drawing && ImGui::IsMouseDown(0) && hovered && stroke.size() < 4096) {
		auto point = world(io.MousePos);
		if(glm::length(point - stroke.back()) > mapSpan * 0.005f) stroke.push_back(point);
	}
	if(ImGui::IsMouseReleased(0)) {
		mapDragKey = -1;
		if(drawing) {
			drawing = false;
			std::vector<float> lengths(stroke.size(), 0);
			for(size_t i = 1; i < stroke.size(); ++i) lengths[i] = lengths[i - 1] + glm::length(stroke[i] - stroke[i - 1]);
			if(stroke.size() > 1 && lengths.back() > 1e-5f) {
				std::vector<glm::vec3> positions;
				for(int i = 0; i < generatorCount; ++i) {
					float distance = lengths.back() * i / (generatorCount - 1);
					size_t end = std::clamp<size_t>(size_t(std::lower_bound(lengths.begin(), lengths.end(), distance) - lengths.begin()), 1, stroke.size() - 1);
					float u = (distance - lengths[end - 1]) / std::max(lengths[end] - lengths[end - 1], 1e-6f);
					positions.push_back(glm::mix(stroke[end - 1], stroke[end], u));
				}
				generate(std::move(positions), false);
			} else status = "Draw a longer stroke to generate a path.";
			stroke.clear();
		}
	}
	if(hovered && ImGui::IsMouseDragging(2)) {
		mapCenter[axisX] -= io.MouseDelta.x / scale;
		mapCenter[axisY] += io.MouseDelta.y / scale;
	}
	if(hovered && io.MouseWheel != 0)
		mapSpan = std::clamp(mapSpan * std::pow(0.85f, io.MouseWheel), 0.001f, 1e7f);
}

void CameraStudio::Impl::drawOverlay() {
	if(!showFrustums || editingMain || (followMain && timelinePreview)) return;
	auto draw = ImGui::GetBackgroundDrawList();
	auto line = [&](glm::vec3 a, glm::vec3 b, ImU32 color, float thickness) {
		ImVec2 sa, sb;
		if(projectMain(a, sa) && projectMain(b, sb)) draw->AddLine(sa, sb, color, thickness);
	};
	if(path.segmentCount() > 0) {
		int steps = std::min(2048, path.segmentCount() * 24);
		for(int i = 1; i <= steps; ++i)
			line(path.sample(path.duration() * (i - 1) / steps).position,
				path.sample(path.duration() * i / steps).position, IM_COL32(140, 175, 255, 220), 2);
	}
	float aspect = float(outputWidth) / outputHeight;
	for(int i = 0; i < int(path.keys.size()); ++i) {
		auto& key = path.keys[i]; float depth = markerSize;
		float h = depth * std::tan(glm::radians(key.fovy) * 0.5f), w = h * aspect;
		std::array<glm::vec3, 4> corners = {glm::vec3(-w, -h, -depth), {w, -h, -depth}, {w, h, -depth}, {-w, h, -depth}};
		for(auto& point : corners) point = glm::vec3(key.world() * glm::vec4(point, 1));
		ImU32 color = i == selected ? IM_COL32(255, 200, 70, 255) : IM_COL32(155, 190, 255, 220);
		for(int j = 0; j < 4; ++j) {
			line(key.position, corners[j], color, 1); line(corners[j], corners[(j + 1) % 4], color, 1.5f);
		}
		ImVec2 screen;
		if(projectMain(key.position, screen)) {
			draw->AddCircleFilled(screen, i == selected ? 6 : 4, color);
			std::string label = std::to_string(i + 1);
			draw->AddText({screen.x + 8, screen.y - 8}, color, label.c_str());
		}
	}
	if(timelinePreview || recording) {
		ImVec2 position;
		if(projectMain(path.sample(playhead).position, position))
			draw->AddCircle(position, 10, IM_COL32(100, 245, 170, 255), 16, 2);
	}
}

void CameraStudio::Impl::savePath() {
	try {
		Json document = {{"format", "splatshop-camera-path"}, {"version", 1},
			{"closed", path.closed}, {"smooth", path.smooth},
			{"output", {{"width", outputWidth}, {"height", outputHeight}, {"fps", outputFps}, {"jpeg_quality", jpegQuality}}}};
		document["cameras"] = Json::array();
		for(auto& key : path.keys) document["cameras"].push_back({
			{"position", {key.position.x, key.position.y, key.position.z}},
			{"quaternion_xyzw", {key.orientation.x, key.orientation.y, key.orientation.z, key.orientation.w}},
			{"fov_y_degrees", key.fovy}, {"seconds_to_next", key.secondsToNext}});
		std::ofstream output(fs::u8path(pathFile));
		if(!output || !(output << document.dump(2) << '\n')) throw std::runtime_error("Cannot save camera path.");
		output.flush();
		if(!output) throw std::runtime_error("Cannot finish saving camera path.");
		status = "Camera path saved.";
	} catch(const std::exception& error) { status = error.what(); }
}

void CameraStudio::Impl::loadPath() {
	try {
		std::ifstream input(fs::u8path(pathFile));
		if(!input) throw std::runtime_error("Cannot open camera path.");
		if(fs::file_size(fs::u8path(pathFile)) > 4 * 1024 * 1024) throw std::runtime_error("Camera path file is too large.");
		Json document; input >> document;
		if(document.value("format", "") != "splatshop-camera-path" || document.value("version", 0) != 1)
			throw std::runtime_error("Unsupported camera path format.");
		CameraPath loaded; loaded.closed = document.value("closed", false); loaded.smooth = document.value("smooth", true);
		auto& cameras = document.at("cameras");
		if(!cameras.is_array() || cameras.size() > 512) throw std::runtime_error("Invalid camera count (maximum 512).");
		for(auto& item : cameras) {
			auto position = item.at("position").get<std::array<float, 3>>();
			auto q = item.at("quaternion_xyzw").get<std::array<float, 4>>();
			CameraKeyframe key; key.position = {position[0], position[1], position[2]};
			key.orientation = {q[3], q[0], q[1], q[2]};
			key.fovy = item.at("fov_y_degrees").get<float>(); key.secondsToNext = item.at("seconds_to_next").get<float>();
			float norm = cameraQuaternionLength(key.orientation);
			if(!finite(key.position) || !std::isfinite(norm) || norm < 1e-5f ||
				!std::isfinite(key.fovy) || key.fovy < 5 || key.fovy > 150 ||
				!std::isfinite(key.secondsToNext) || key.secondsToNext < 0.05f || key.secondsToNext > 600)
				throw std::runtime_error("Invalid camera pose, FOV or segment duration.");
			key.orientation = cameraNormalizeQuaternion(key.orientation); loaded.keys.push_back(key);
		}
		auto output = document.value("output", Json::object());
		int width = output.value("width", 1280), height = output.value("height", 720);
		int fps = output.value("fps", 30), quality = output.value("jpeg_quality", 90);
		if(width < 16 || height < 16 || width > 4096 || height > 4096 || fps < 1 || fps > 120 || quality < 1 || quality > 100)
			throw std::runtime_error("Invalid camera path output settings.");
		// 全部校验通过再替换当前轨迹，读取失败时保留编辑内容。
		checkpoint(); path = std::move(loaded); selected = path.keys.empty() ? -1 : 0;
		outputWidth = width; outputHeight = height; outputFps = fps; jpegQuality = quality;
		playing = false; playhead = 0; changed(); status = "Camera path loaded.";
	} catch(const std::exception& error) { status = error.what(); }
}

void CameraStudio::Impl::startRecording(SplatEditor& editor) {
	try {
		if(path.segmentCount() == 0) throw std::runtime_error("Add at least two cameras before recording.");
		if(editor.ovr->isActive() || editor.settings.multiView || editor.settings.enableStereoFramebufferTest)
			throw std::runtime_error("Use the single desktop view for Camera Studio recording.");
		auto destination = fs::u8path(videoFile);
		auto extension = destination.extension().string();
		std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char c) { return char(std::tolower(c)); });
		if(extension != ".avi") throw std::runtime_error("Use an .avi output filename.");
		if(!overwriteVideo && fs::exists(destination)) throw std::runtime_error("Output already exists. Change the filename or enable overwrite.");
		editingMain = false; restoreOverview();
		video.open(destination, outputWidth, outputHeight, outputFps);
		totalFrames = std::max(2u, uint32_t(std::ceil(path.duration() * outputFps)));
		recordingWidth = outputWidth; recordingHeight = outputHeight; recordingQuality = jpegQuality;
		recording = true; playing = false; timelinePreview = true; playhead = 0;
		previewOpen = true; status = "Recording AVI (MJPEG), one rendered frame at a time.";
	} catch(const std::exception& error) { status = error.what(); }
}

void CameraStudio::Impl::stopRecording(bool completed) {
	uint32_t frames = video.frameCount();
	bool success = video.close(); recording = playing = false;
	status = success ? (completed ? "Video saved: " : "Recording stopped; partial video saved: ") +
		std::to_string(frames) + " frames." : "Video finalization failed; check available disk space.";
}

void CameraStudio::Impl::drawEditor(SplatEditor& editor) {
	ImGui::SetNextWindowPos({20, 130}, ImGuiCond_FirstUseEver);
	ImGui::SetNextWindowSize({450, 730}, ImGuiCond_FirstUseEver);
	if(!ImGui::Begin("Camera Trajectory", &editorOpen)) { ImGui::End(); return; }
	ImGui::TextWrapped("Draw a path, select cameras, then refine them in the main scene.");
	if(ImGui::Button("Camera preview")) previewOpen = true;
	ImGui::SameLine(); ImGui::Text("%zu cameras | %.2f seconds", path.keys.size(), path.duration());
	if(recording) {
		ImGui::ProgressBar(float(video.frameCount()) / totalFrames, {-1, 0});
		if(ImGui::Button("Stop recording")) stopRecording(false);
	}
	{
		DisabledScope disabled(recording || editingMain);
		if(ImGui::Button("Add current view")) addCurrent(false);
		ImGui::SameLine(); if(ImGui::Button("Insert after selected")) addCurrent(true);
		{
			DisabledScope unavailable(undoStack.empty());
			if(ImGui::Button("Undo path")) history(false);
		}
		ImGui::SameLine(); {
			DisabledScope unavailable(redoStack.empty());
			if(ImGui::Button("Redo path")) history(true);
		}
		ImGui::SameLine(); if(ImGui::Button("Clear path") && !path.keys.empty()) {
			checkpoint(); path.keys.clear(); selected = -1; playing = false; changed();
		}
		ImGui::SetNextItemWidth(120);
		if(ImGui::Combo("Plane", &plane, "XY (top)\0XZ (front)\0YZ (side)\0"))
			planeOffset = currentCamera().position[plane == 0 ? 2 : (plane == 1 ? 1 : 0)];
		ImGui::SameLine(); ImGui::SetNextItemWidth(120); ImGui::Combo("Mode", &mapMode, "Select / drag\0Add camera\0Draw stroke\0");
		ImGui::SetNextItemWidth(130); ImGui::DragFloat("Plane coordinate", &planeOffset, 0.01f);
		ImGui::SameLine(); if(ImGui::Button("Refresh map")) refreshMap(editor);
		if(ImGui::Button("Fit path")) {
			if(!path.keys.empty()) {
				glm::vec3 low = path.keys.front().position, high = low;
				for(auto& key : path.keys) { low = glm::min(low, key.position); high = glm::max(high, key.position); }
				mapCenter = (low + high) * 0.5f; mapSpan = std::max(glm::length(high - low) * 1.4f, 0.1f);
			}
		}
		ImGui::SameLine(); if(ImGui::Button("Center on main target")) mapCenter = glm::vec3(Runtime::controls->target);
		drawMap();
		ImGui::TextDisabled("Click camera to select; drag to move. Middle drag: pan; wheel: zoom.");
		if(ImGui::CollapsingHeader("Generate path", ImGuiTreeNodeFlags_DefaultOpen)) {
			ImGui::SetNextItemWidth(180); ImGui::Combo("Camera aim", &aimMode, "Look at target\0Follow path\0Keep current orientation\0");
			ImGui::DragFloat3("Aim target", &aimTarget.x, 0.01f);
			if(ImGui::Button("Use main orbit target")) aimTarget = glm::vec3(Runtime::controls->target);
			ImGui::SliderInt("Generated cameras", &generatorCount, 2, 128, "%d", ImGuiSliderFlags_AlwaysClamp);
			ImGui::Checkbox("Append generated cameras", &appendGenerated);
			ImGui::DragFloat("Orbit radius", &orbitRadius, 0.01f, 0.001f, 1e6f, "%.3f", ImGuiSliderFlags_AlwaysClamp);
			ImGui::DragFloat("Orbit height", &orbitHeight, 0.01f);
			ImGui::SliderFloat("Orbit arc", &orbitArc, 10, 360, "%.0f degrees", ImGuiSliderFlags_AlwaysClamp);
			if(ImGui::Button("Generate orbit")) {
				std::vector<glm::vec3> positions;
				bool closed = orbitArc >= 359.9f;
				for(int i = 0; i < generatorCount; ++i) {
					float angle = glm::radians(orbitArc) * i / (closed ? generatorCount : generatorCount - 1);
					positions.push_back(aimTarget + glm::vec3(std::cos(angle) * orbitRadius, std::sin(angle) * orbitRadius, orbitHeight));
				}
				generate(std::move(positions), closed);
			}
			ImGui::SameLine(); if(ImGui::Button("Straighten path") && path.keys.size() > 2) {
				checkpoint(); auto start = path.keys.front().position, end = path.keys.back().position;
				for(size_t i = 0; i < path.keys.size(); ++i) path.keys[i].position = glm::mix(start, end, float(i) / (path.keys.size() - 1));
				path.closed = false; changed();
			}
		}
		if(ImGui::BeginChild("Camera list", {0, 115}, true)) {
			for(int i = 0; i < int(path.keys.size()); ++i) {
				auto& key = path.keys[i];
				std::string label = std::format("Camera {:03} | ({:.2f}, {:.2f}, {:.2f})", i + 1, key.position.x, key.position.y, key.position.z);
				if(ImGui::Selectable(label.c_str(), selected == i)) { select(i); previewOpen = true; }
			}
		}
		ImGui::EndChild();
		{
			DisabledScope unavailable(!validSelection());
			if(ImGui::Button("Delete camera")) {
				checkpoint(); path.keys.erase(path.keys.begin() + selected);
				selected = path.keys.empty() ? -1 : std::min(selected, int(path.keys.size()) - 1); changed();
			}
			ImGui::SameLine(); if(ImGui::Button("Duplicate") && path.keys.size() < 512) {
				checkpoint(); auto key = path.keys[selected]; path.keys.insert(path.keys.begin() + selected + 1, key); ++selected; changed();
			}
			ImGui::SameLine(); if(ImGui::Button("Earlier") && selected > 0) {
				checkpoint(); std::swap(path.keys[selected], path.keys[selected - 1]); --selected; changed();
			}
			ImGui::SameLine(); if(ImGui::Button("Later") && selected + 1 < int(path.keys.size())) {
				checkpoint(); std::swap(path.keys[selected], path.keys[selected + 1]); ++selected; changed();
			}
		}
		bool closed = path.closed, smooth = path.smooth;
		if(ImGui::Checkbox("Closed path", &closed)) { checkpoint(); path.closed = closed; changed(); }
		ImGui::SameLine(); if(ImGui::Checkbox("Smooth position", &smooth)) { checkpoint(); path.smooth = smooth; changed(); }
		if(path.segmentCount() > 0) {
			float duration = path.duration();
			bool modified = ImGui::DragFloat("Path duration", &duration, 0.1f, 0.1f, 3600, "%.2f s", ImGuiSliderFlags_AlwaysClamp);
			itemCheckpoint();
			if(modified) {
				float factor = duration / std::max(path.duration(), 0.01f);
				for(int i = 0; i < path.segmentCount(); ++i)
					path.keys[i].secondsToNext = std::clamp(path.keys[i].secondsToNext * factor, 0.05f, 600.0f);
				changed();
			}
		}
	}
	ImGui::Separator();
	{
		DisabledScope disabled(recording || editingMain || path.segmentCount() == 0);
		if(ImGui::Button(playing ? "Pause" : "Play path")) {
			if(!playing && playhead >= path.duration()) playhead = 0;
			playing = !playing; timelinePreview = true;
		}
		ImGui::SameLine(); if(ImGui::Button("Stop / overview")) {
			playing = timelinePreview = false; playhead = 0; restoreOverview();
		}
		ImGui::SameLine(); ImGui::Checkbox("Repeat", &loopPlayback);
		if(ImGui::SliderFloat("Timeline", &playhead, 0, std::max(path.duration(), 0.01f), "%.2f s")) {
			playing = false; timelinePreview = true;
		}
		ImGui::SliderFloat("Playback speed", &playbackSpeed, 0.1f, 3, "%.2fx", ImGuiSliderFlags_AlwaysClamp);
		ImGui::Checkbox("Follow path in main view", &followMain);
	}
	{
		DisabledScope disabled(recording || editingMain);
		if(ImGui::CollapsingHeader("Save / load path")) {
			ImGui::InputText("Path JSON", pathFile, sizeof(pathFile));
			if(ImGui::Button("Save path")) savePath();
			ImGui::SameLine(); if(ImGui::Button("Load path")) loadPath();
		}
		if(ImGui::CollapsingHeader("Record video", ImGuiTreeNodeFlags_DefaultOpen)) {
			ImGui::InputInt("Width", &outputWidth); outputWidth = std::clamp(outputWidth, 16, 4096);
			ImGui::InputInt("Height", &outputHeight); outputHeight = std::clamp(outputHeight, 16, 4096);
			ImGui::InputInt("FPS", &outputFps); outputFps = std::clamp(outputFps, 1, 120);
			ImGui::SliderInt("JPEG quality", &jpegQuality, 1, 100);
			ImGui::InputText("Video AVI", videoFile, sizeof(videoFile));
			ImGui::Checkbox("Overwrite existing video", &overwriteVideo);
			if(ImGui::Button("Record whole path")) startRecording(editor);
			ImGui::TextDisabled("AVI / MJPEG, no audio. UI, camera markers and grid are excluded.");
		}
		ImGui::Checkbox("Show path and cameras in scene", &showFrustums);
		ImGui::DragFloat("Camera marker size", &markerSize, 0.001f, 0.001f, 1000, "%.3f", ImGuiSliderFlags_AlwaysClamp);
	}
	if(!status.empty()) ImGui::TextWrapped("%s", status.c_str());
	ImGui::End();
}

void CameraStudio::Impl::drawPreview(SplatEditor& editor) {
	ImGui::SetNextWindowPos({float(std::max(490, GLRenderer::width - 530)), 130}, ImGuiCond_FirstUseEver);
	ImGui::SetNextWindowSize({510, 600}, ImGuiCond_FirstUseEver);
	previewVisible = ImGui::Begin("Camera Preview", &previewOpen);
	if(!previewVisible) { ImGui::End(); return; }
	if(timelinePreview || recording) ImGui::Text("Path preview | %.2f / %.2f s", playhead, path.duration());
	else if(validSelection()) ImGui::Text("Selected camera: %d | %.2f s", selected + 1, path.keyTime(selected));
	else ImGui::TextWrapped("Add a camera from the main view, draw a stroke, or generate an orbit.");
	if(recording) {
		ImGui::ProgressBar(float(video.frameCount()) / totalFrames, {-1, 0});
		if(ImGui::Button("Stop recording##preview")) stopRecording(false);
	}
	float aspect = float(outputWidth) / outputHeight;
	float width = ImGui::GetContentRegionAvail().x;
	float height = std::min(width / aspect, 310.0f);
	width = height * aspect;
	if(previewFramebuffer) {
		ImVec2 imageStart = ImGui::GetCursorScreenPos();
		ImGui::Image((ImTextureID)(intptr_t)previewFramebuffer->colorAttachments[0]->handle,
			{width, height}, {0, 1}, {1, 0});
		ImVec2 afterImage = ImGui::GetCursorScreenPos();
		if(validSelection() && !playing && !recording && !timelinePreview) {
			ImGui::SetCursorScreenPos(imageStart); ImGui::InvisibleButton("Aim camera in preview", {width, height});
			if(ImGui::IsItemHovered() && ImGui::IsMouseClicked(0)) { checkpoint(); previewDragging = true; }
			if(previewDragging && ImGui::IsMouseDown(0)) {
				auto delta = ImGui::GetIO().MouseDelta;
				auto yaw = cameraAngleAxis(-delta.x * 0.003f, glm::vec3(0, 0, 1));
				auto pitch = cameraAngleAxis(-delta.y * 0.003f, glm::vec3(1, 0, 0));
				auto& key = path.keys[selected]; key.orientation = cameraNormalizeQuaternion(yaw * key.orientation * pitch);
			}
			if(ImGui::IsMouseReleased(0)) previewDragging = false;
			ImGui::SetCursorScreenPos(afterImage);
		}
	} else ImGui::Dummy({width, height});
	ImGui::TextDisabled("Drag the selected camera preview to adjust its aim.");
	{
		DisabledScope disabled(recording || playing || !validSelection());
		if(ImGui::Button(editingMain ? "Back to overview" : "Edit selected in main view")) {
			if(editingMain) { editingMain = false; restoreOverview(); }
			else {
				restoreOverview(); saveOverview(); editingMain = true;
				timelinePreview = false; editor.setAction(nullptr); applyMainCamera(path.keys[selected]); consumeMouse();
			}
		}
		if(editingMain) {
			ImGui::TextWrapped("Main view: left drag to aim; right drag to pan; wheel / WASD to move; Q/E up/down. Shift speeds up.");
			ImGui::DragFloat("Move speed", &moveSpeed, 0.01f, 0.001f, 1e5f, "%.3f", ImGuiSliderFlags_AlwaysClamp);
		} else {
			ImGui::TextWrapped("Overview: click a camera marker, then use its move / rotate gizmo.");
			if(ImGui::RadioButton("Move camera", operation == ImGuizmo::TRANSLATE)) operation = ImGuizmo::TRANSLATE;
			ImGui::SameLine(); if(ImGui::RadioButton("Rotate camera", operation == ImGuizmo::ROTATE)) operation = ImGuizmo::ROTATE;
			ImGui::Checkbox("Local gizmo", &localGizmo);
		}
		if(validSelection()) {
			auto& key = path.keys[selected];
			auto position = key.position;
			bool modified = ImGui::DragFloat3("Position", &position.x, 0.01f);
			itemCheckpoint();
			if(modified) { key.position = position; changed(); }
			glm::mat4 matrix = key.world(); glm::vec3 translation, rotation, scale;
			ImGuizmo::DecomposeMatrixToComponents(&matrix[0][0], &translation.x, &rotation.x, &scale.x);
			modified = ImGui::DragFloat3("Rotation XYZ", &rotation.x, 0.25f); itemCheckpoint();
			if(modified) {
				glm::vec3 unitScale(1);
				ImGuizmo::RecomposeMatrixFromComponents(&translation.x, &rotation.x, &unitScale.x, &matrix[0][0]);
				key.orientation = cameraNormalizeQuaternion(glm::quat_cast(glm::mat3(matrix))); changed();
			}
			float fovy = key.fovy, seconds = key.secondsToNext;
			modified = ImGui::SliderFloat("Vertical FOV", &fovy, 5, 150, "%.2f degrees"); itemCheckpoint();
			if(modified) { key.fovy = std::clamp(fovy, 5.0f, 150.0f); changed(); }
			modified = ImGui::DragFloat("Seconds to next camera", &seconds, 0.05f, 0.05f, 600, "%.2f s"); itemCheckpoint();
			if(modified) { key.secondsToNext = std::clamp(seconds, 0.05f, 600.0f); changed(); }
			if(ImGui::Button("Use current main view")) {
				checkpoint(); float seconds = key.secondsToNext; key = currentCamera(); key.secondsToNext = seconds; changed();
			}
			ImGui::SameLine(); if(ImGui::Button("Aim at target")) { checkpoint(); key.aimAt(aimTarget); changed(); }
			if(ImGui::Button("Preview from this camera")) { playing = false; timelinePreview = false; }
		}
	}
	if(!status.empty()) ImGui::TextWrapped("%s", status.c_str());
	ImGui::End();
}

void CameraStudio::drawGui(SplatEditor& editor) {
	auto& s = *impl;
	if(!s.active() || editor.ovr->isActive()) return;
	if(editor.settings.multiView || editor.settings.enableStereoFramebufferTest) {
		if(s.recording) s.stopRecording(false);
		s.status = "Switch off Four Views and Stereo Framebuffer Test to edit camera poses.";
	} else s.drawOverlay();
	if(s.editorOpen) s.drawEditor(editor);
	if(s.previewOpen) s.drawPreview(editor);
	else { s.previewVisible = false; s.editingMain = false; }
	if(!s.editorOpen && !s.previewOpen) {
		if(s.recording) s.stopRecording(false);
		s.playing = s.editingMain = false; s.restoreOverview();
	}
}
