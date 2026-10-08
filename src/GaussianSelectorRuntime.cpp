#ifdef SPLATSHOP_ENABLE_GAUSSIAN_SELECTOR

#include "GaussianSelectorRuntime.h"

#include "SceneSnapshotAdapter.h"
#include "SplatEditor.h"
#include "actions/InputAction.h"
#include "gaussian_selector/CanonicalAxisColor.h"
#include "gaussian_selector/LeidenAdapter.h"
#include "gaussian_selector/SeedEvidence.h"
#include "gaussian_selector/SpatialGraph.h"
#include "gaussian_selector/ScreenRoi.h"
#include "gaussian_selector/SuperpointEncoding.h"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <thread>

using namespace gaussian_selector;

namespace {

uint64_t hashBytes(uint64_t hash, const void* data, std::size_t bytes) {
	const auto* p = static_cast<const uint8_t*>(data);
	for(std::size_t i = 0; i < bytes; ++i) {
		hash ^= p[i];
		hash *= 1099511628211ull;
	}
	return hash;
}

uint64_t configHash(const SelectorConfig& c) {
	uint64_t h = 1469598103934665603ull;
	h = hashBytes(h, &c.knn, sizeof(c.knn));
	h = hashBytes(h, &c.gateQuantile, sizeof(c.gateQuantile));
	h = hashBytes(h, &c.spatialWeight, sizeof(c.spatialWeight));
	h = hashBytes(h, &c.colorWeight, sizeof(c.colorWeight));
	h = hashBytes(h, &c.opacityWeight, sizeof(c.opacityWeight));
	h = hashBytes(h, &c.leidenResolution, sizeof(c.leidenResolution));
	h = hashBytes(h, &c.leidenBeta, sizeof(c.leidenBeta));
	h = hashBytes(h, &c.leidenIterations, sizeof(c.leidenIterations));
	h = hashBytes(h, &c.randomSeed, sizeof(c.randomSeed));
	h = hashBytes(h, &c.graphNodeTarget, sizeof(c.graphNodeTarget));
	h = hashBytes(h, &c.graphNodeBudget, sizeof(c.graphNodeBudget));
	h = hashBytes(h, &c.voxelSearchIterations, sizeof(c.voxelSearchIterations));
	return h;
}

std::size_t checkedMultiply(std::size_t a, std::size_t b, const char* label) {
	if(a && b > std::numeric_limits<std::size_t>::max() / a)
		throw std::overflow_error(std::string(label) + " size overflow");
	return a * b;
}

std::size_t checkedAdd(std::size_t a, std::size_t b, const char* label) {
	if(b > std::numeric_limits<std::size_t>::max() - a)
		throw std::overflow_error(std::string(label) + " size overflow");
	return a + b;
}

struct GaussianSelectorSelectionUndoAction : Action {
	std::shared_ptr<SNSplats> node;
	uint64_t indexEpoch = 0;
	uint32_t count = 0;
	CUdeviceptr swapMask = 0;

	~GaussianSelectorSelectionUndoAction() {
		if(swapMask) CURuntime::free(swapMask);
	}

	void swap() {
		if(!node || node->indexEpoch != indexEpoch || node->dmng.data.count != count) {
			println("GaussianSelector undo refused: stale node/index epoch");
			return;
		}
		auto* editor = SplatEditor::instance;
		editor->prog_gaussians_editing->launch(
			"kernel_swap_selection", {&node->dmng.data, &swapMask}, count);
	}

	void undo() override { swap(); }
	void redo() override { swap(); }
	int64_t byteSize() override { return swapMask ? CURuntime::getSize(swapMask) : 0; }
};

} // namespace

const char* gaussianSelectorStateName(GaussianSelectorState state) {
	switch(state) {
		case GaussianSelectorState::Idle: return "Idle";
		case GaussianSelectorState::Snapshotting: return "Snapshotting";
		case GaussianSelectorState::Encoding: return "Encoding";
		case GaussianSelectorState::Lifting: return "Lifting";
		case GaussianSelectorState::Solving: return "Solving";
		case GaussianSelectorState::PreviewReady: return "PreviewReady";
		case GaussianSelectorState::Cancelled: return "Cancelled";
		case GaussianSelectorState::Failed: return "Failed";
		case GaussianSelectorState::Stale: return "Stale";
	}
	return "Unknown";
}

GaussianSelectorRuntime::GaussianSelectorRuntime(SplatEditor* editor) : editor_(editor) {
	cuEventCreate(&coverageDone_, CU_EVENT_DISABLE_TIMING);
}

GaussianSelectorRuntime::~GaussianSelectorRuntime() {
	cancel();
	releaseDeviceResources();
	if(coverageDone_) cuEventDestroy(coverageDone_);
	if(hMassPinned_) cuMemFreeHost(hMassPinned_);
}

uint64_t GaussianSelectorRuntime::viewKey(int width, int height, const mat4& view, const mat4& proj) {
	uint64_t h = 1469598103934665603ull;
	h = hashBytes(h, &width, sizeof(width));
	h = hashBytes(h, &height, sizeof(height));
	h = hashBytes(h, &view, sizeof(view));
	return hashBytes(h, &proj, sizeof(proj));
}

uint64_t GaussianSelectorRuntime::visibilityRevision() const {
	uint64_t h = 1469598103934665603ull;
	const auto& settings = editor_->settings;
	h = hashBytes(h, &settings.splatSize, sizeof(settings.splatSize));
	h = hashBytes(h, &settings.cullSmallSplats, sizeof(settings.cullSmallSplats));
	h = hashBytes(h, &settings.makePoints, sizeof(settings.makePoints));
	h = hashBytes(h, &settings.showRing, sizeof(settings.showRing));
	h = hashBytes(h, &settings.intersectionMode, sizeof(settings.intersectionMode));
	editor_->scene.root->traverse([&](SceneNode* node) {
		h = hashBytes(h, &node->ID, sizeof(node->ID));
		h = hashBytes(h, &node->visible, sizeof(node->visible));
		h = hashBytes(h, &node->transform_global, sizeof(node->transform_global));
		if(auto* splats = dynamic_cast<SNSplats*>(node)) {
			h = hashBytes(h, &splats->indexEpoch, sizeof(splats->indexEpoch));
			h = hashBytes(h, &splats->contentRevision, sizeof(splats->contentRevision));
			h = hashBytes(h, &splats->dmng.data.count, sizeof(splats->dmng.data.count));
			h = hashBytes(h, &splats->dmng.data.writeDepth, sizeof(splats->dmng.data.writeDepth));
		} else if(auto* triangles = dynamic_cast<SNTriangles*>(node)) {
			h = hashBytes(h, &triangles->data.count, sizeof(triangles->data.count));
			h = hashBytes(h, &triangles->data.position, sizeof(triangles->data.position));
		} else if(auto* points = dynamic_cast<SNPoints*>(node)) {
			h = hashBytes(h, &points->manager.data.count, sizeof(points->manager.data.count));
			h = hashBytes(h, &points->manager.data.position, sizeof(points->manager.data.position));
		}
	});
	return h;
}

uint64_t GaussianSelectorRuntime::evidenceViewKey(int width, int height,
	const mat4& view, const mat4& proj) const {
	const uint64_t visibility = visibilityRevision();
	return hashBytes(viewKey(width, height, view, proj), &visibility, sizeof(visibility));
}

void GaussianSelectorRuntime::ensureDevice(CUdeviceptr& pointer, std::size_t& capacity,
	std::size_t requiredBytes, const char* label) {
	if(requiredBytes <= capacity) return;
	if(pointer) CURuntime::free(pointer);
	pointer = requiredBytes ? CURuntime::alloc(label, requiredBytes) : 0;
	capacity = requiredBytes;
}

void GaussianSelectorRuntime::releaseDeviceResources() {
	for(auto* p : {&dStageRows_, &dLocalToSnapshot_, &dInitialDepth_, &dForegroundMask_, &dBackgroundMask_,
		&dAllMass_, &dForegroundMass_, &dBackgroundMass_, &dSupportCenter_, &dPreviewWords_}) {
		if(*p) CURuntime::free(*p);
		*p = 0;
	}
	stageRowsCapacity_ = localToSnapshotCapacity_ = depthCapacity_ = foregroundMaskCapacity_ = backgroundMaskCapacity_ = 0;
	allMassCapacity_ = foregroundMassCapacity_ = backgroundMassCapacity_ = previewCapacity_ = 0;
	supportCenterCapacity_=0;
}

void GaussianSelectorRuntime::fail(std::string message) {
	state_ = GaussianSelectorState::Failed;
	message_ = std::move(message);
	coverageRequested_ = coveragePrepared_ = coverageInFlight_ = false;
	solveInFlight_ = false;
}

bool GaussianSelectorRuntime::snapshotStillValid() const {
	return target_ && target_->indexEpoch == snapshotKey_.indexEpoch &&
		target_->contentRevision == targetContentRevision_ &&
		SceneSnapshotAdapter::transformRevision(*target_) == targetTransformRevision_ &&
		target_->dmng.data.count == snapshotNodeCount_;
}

void GaussianSelectorRuntime::captureSnapshot(const std::shared_ptr<SNSplats>& target) {
	const auto& data = target->dmng.data;
	const std::size_t n = data.count;
	if(n == 0) throw std::runtime_error("selected Gaussian layer is empty");
	if(data.numSHCoefficients < 0) throw std::runtime_error("invalid negative SH coefficient count");
	std::function<bool(glm::vec3)> include;
	if(screenRoi_) {
		const auto box=*screenRoi_;
		const ScreenRoi roi{box.x,box.y,box.z,box.w};
		const auto vp=requestProj_*requestView_;
		const int width=requestWidth_,height=requestHeight_;
		include=[roi,vp,width,height](glm::vec3 world) {
			const auto clip=vp*glm::vec4(world,1);
			return roi.containsClip(clip.x,clip.y,clip.z,clip.w,width,height);
		};
	}
	// Reserve encoding workspace in the same preflight as the selected snapshot.
	const auto graphBytes=checkedAdd(1024,checkedMultiply(std::size_t(config_.knn),
		8*sizeof(WeightedEdge),"ROI graph workspace"),"ROI graph workspace");
	const auto graphReserve=checkedMultiply(config_.graphNodeBudget,graphBytes,"bounded graph workspace");
	if(graphReserve>=config_.cpuBudgetBytes)throw std::runtime_error("graph node budget exceeds CPU budget");
	auto captured = SceneSnapshotAdapter::capture(*target,include,config_.cpuBudgetBytes-graphReserve,
		std::size_t(kInvalidRow),256);
	if(captured.rows->empty()) throw std::runtime_error("ROI/layer contains no valid non-deleted finite Gaussian centers");
	snapshotRows_ = std::move(captured.rows);
	localToSnapshot_ = std::move(captured.localToSnapshot);
	snapshotNodeCount_ = captured.nodeCount;
	targetContentRevision_ = target->contentRevision;
	targetTransformRevision_ = captured.transformRevision;
	uint64_t sceneRevision = hashBytes(1469598103934665603ull, &targetContentRevision_, sizeof(targetContentRevision_));
	sceneRevision = hashBytes(sceneRevision, &targetTransformRevision_, sizeof(targetTransformRevision_));
	sceneRevision = hashBytes(sceneRevision, &target->ID, sizeof(target->ID));
	// Cache identity includes the actual compact active-domain content, not
	// only a rectangle/camera proxy. Ordered rows also define mask row identity.
	for(const auto& row:*snapshotRows_)sceneRevision=hashBytes(sceneRevision,&row.identity.localIndex,sizeof(row.identity.localIndex));
	snapshotKey_ = {sceneRevision, configHash(config_), target->indexEpoch};
	message_ = "Snapshot captured; skipped " + std::to_string(captured.skippedDeleted) + " deleted and " + std::to_string(captured.skippedNonFinite) + " non-finite Gaussians";
}

void GaussianSelectorRuntime::startEncoding(uint64_t requestId) {
	state_ = GaussianSelectorState::Encoding;
	message_ = "Building CAC18, budgeted Gaussian/voxel kNN, Leiden and SP graph";
	encodingReady_ = false;
	auto rows = snapshotRows_;
	if(!encodingRowToSnapshot_.empty()) {
		auto scoped = std::make_shared<std::vector<GaussianSnapshotRow>>();
		scoped->reserve(encodingRowToSnapshot_.size());
		for(auto row : encodingRowToSnapshot_) scoped->push_back((*snapshotRows_)[row]);
		rows = std::move(scoped);
	}
	auto key = snapshotKey_;
	auto config = config_;
	auto mailbox = mailbox_;
	std::thread([rows, key, config, mailbox, requestId] {
		EncodingPublication publication;
		publication.requestId = requestId;
		try {
			publication.encoding = std::make_shared<SceneEncoding>(encodeScene(key,*rows,config));
		} catch(const std::exception& e) {
			publication.error = e.what();
		}
		std::scoped_lock lock(mailbox->mutex);
		mailbox->encodings.push_back(std::move(publication));
	}).detach();
}

void GaussianSelectorRuntime::requestSolve(const std::shared_ptr<SNSplats>& target,
	int width, int height, const mat4& view, const mat4& proj,
	const std::vector<uint8_t>& foregroundMask,
	const std::vector<uint8_t>& backgroundMask, std::optional<glm::vec4> screenRoi) {
	requestId_++;
	resetPreview();
	coverageRequested_=coveragePrepared_=coverageInFlight_=coverageReady_=solveInFlight_=false;
	diagnostics_ = {};
	if(!target || target->locked || !target->visible) return fail("select one visible, unlocked Gaussian layer");
	std::size_t requestedPixels = 0;
	try {
		if(width <= 0 || height <= 0) throw std::invalid_argument("invalid viewport dimensions");
		requestedPixels = checkedMultiply(std::size_t(width), std::size_t(height), "viewport");
	} catch(const std::exception& e) {
		return fail(e.what());
	}
	if(foregroundMask.size() != requestedPixels || backgroundMask.size() != foregroundMask.size())
		return fail("stroke mask dimensions do not match the viewport");
	if(!std::isfinite(config_.evidenceTransmittanceStop) || config_.evidenceTransmittanceStop < 0 || config_.evidenceTransmittanceStop >= 1)
		return fail("evidence transmittance stop must be in [0,1)");
	uint64_t evidenceConfig = hashBytes(1469598103934665603ull,
		&config_.evidenceTransmittanceStop, sizeof(config_.evidenceTransmittanceStop));
	evidenceConfig = hashBytes(evidenceConfig, &config_.coverageThreshold, sizeof(config_.coverageThreshold));
	evidenceConfig = hashBytes(evidenceConfig, &config_.minimumVisibleMass, sizeof(config_.minimumVisibleMass));
	evidenceConfig = hashBytes(evidenceConfig, &config_.evidenceOwnerBudget, sizeof(config_.evidenceOwnerBudget));
	evidenceConfig = hashBytes(evidenceConfig, &config_.adaptiveCoverageThreshold, sizeof(config_.adaptiveCoverageThreshold));
	evidenceConfig = hashBytes(evidenceConfig, &config_.coverageQuantile, sizeof(config_.coverageQuantile));
	evidenceConfig = hashBytes(evidenceConfig, &config_.centerDepthTolerance, sizeof(config_.centerDepthTolerance));
	evidenceConfig = hashBytes(evidenceConfig, &config_.minimumScribbleSupport, sizeof(config_.minimumScribbleSupport));
	evidenceConfig = hashBytes(evidenceConfig, &config_.requireCenterEvidence, sizeof(config_.requireCenterEvidence));
	if(evidenceConfig != evidenceConfigRevision_) viewEvidence_.clear();
	evidenceConfigRevision_ = evidenceConfig;
	requestEvidenceStop_ = config_.evidenceTransmittanceStop;
	requestOwnerBudget_ = config_.evidenceOwnerBudget;
	requestCenterTolerance_ = float(config_.centerDepthTolerance);
	if(std::none_of(foregroundMask.begin(), foregroundMask.end(), [](uint8_t v){ return v != 0; }) ||
		(!config_.allowPseudoBackground && std::none_of(backgroundMask.begin(), backgroundMask.end(), [](uint8_t v){ return v != 0; }))) {
		return fail("foreground stroke required; add BG or enable pseudo-BG");
	}

	const bool snapshotChanged = !target_ || target_.get() != target.get() ||
		target->indexEpoch != snapshotKey_.indexEpoch ||
		target->contentRevision != targetContentRevision_ ||
		SceneSnapshotAdapter::transformRevision(*target) != targetTransformRevision_ ||
		configHash(config_) != snapshotKey_.encodingRevision;
	if(snapshotChanged) {
		viewEvidence_.clear();
		encoding_.reset();
		encodingRowToSnapshot_.clear();
		roiActive_ = false;
		roiRevision_++;
	}
	target_ = target;
	requestWidth_ = width;
	requestHeight_ = height;
	requestView_ = view;
	requestProj_ = proj;
	const bool roiChanged = screenRoi_ != screenRoi || (screenRoi &&
		pendingViewKey_ != evidenceViewKey(width, height, view, proj));
	screenRoi_ = screenRoi;
	if(roiChanged) {
		encoding_.reset(); encodingRowToSnapshot_.clear(); encodingReady_ = false;
		viewEvidence_.clear(); ++roiRevision_;
	}
	const uint64_t currentVisibilityRevision = visibilityRevision();
	if(evidenceVisibilityRevision_ && evidenceVisibilityRevision_ != currentVisibilityRevision) viewEvidence_.clear();
	evidenceVisibilityRevision_ = currentVisibilityRevision;
	pendingViewKey_ = hashBytes(viewKey(width, height, view, proj),
		&currentVisibilityRevision, sizeof(currentVisibilityRevision));
	requestForegroundMask_ = foregroundMask;
	requestBackgroundMask_ = backgroundMask;
	coverageRequested_ = true;
	coveragePrepared_ = coverageInFlight_ = coverageReady_ = false;
	solveInFlight_ = false;
	state_ = GaussianSelectorState::Snapshotting;
	message_ = "Capturing immutable Gaussian snapshot";
	try {
		if(snapshotChanged || roiChanged || !snapshotRows_) {
			// A compact snapshot belongs to its ROI and camera. Never reuse it
			// after expanding/moving the rectangle, including a failed capture.
			snapshotRows_.reset(); localToSnapshot_.clear();
			captureSnapshot(target);
		}
		if(screenRoi_ || snapshotRows_->size() > config_.fullScopeWarningCount) {
			encodingReady_ = false;
			state_ = GaussianSelectorState::Lifting;
		message_ = "Lifting full-scene visibility before encoding the ROI";
		} else if(snapshotChanged || !encoding_) {
			startEncoding(requestId_);
		} else {
			encodingReady_ = true;
			state_ = GaussianSelectorState::Lifting;
			message_ = "Reusing cached graph; waiting for view evidence";
		}
	} catch(const std::exception& e) {
		fail(e.what());
	}
}

bool GaussianSelectorRuntime::prepareStandardTarget(RenderTarget& target, uint32_t potentialVisibleSplats) {
	if(!coverageRequested_ || coverageInFlight_ || !target_ || target.width != requestWidth_ || target.height != requestHeight_) return false;
	if(evidenceViewKey(target.width,target.height,target.view,target.proj)!=pendingViewKey_) return fail("camera, occlusion scene, or visibility settings changed before evidence capture; press Solve again"),false;
	std::size_t pixels = 0;
	try {
		pixels = checkedMultiply(std::size_t(target.width), std::size_t(target.height), "viewport");
	} catch(const std::overflow_error& e) {
		return fail(e.what()), false;
	}
	const std::size_t n = snapshotRows_->size();
	try {
		std::size_t gpuBytes = checkedMultiply(std::size_t(potentialVisibleSplats), sizeof(uint32_t), "stage identity");
		gpuBytes = checkedAdd(gpuBytes, checkedMultiply(localToSnapshot_.size(), sizeof(uint32_t), "local identity"), "selector GPU");
		gpuBytes = checkedAdd(gpuBytes, checkedMultiply(pixels, sizeof(float) + 2 * sizeof(uint8_t), "view masks/depth"), "selector GPU");
		gpuBytes = checkedAdd(gpuBytes, checkedMultiply(n, 7 * sizeof(float), "coverage mass/support/center"), "selector GPU");
		gpuBytes = checkedAdd(gpuBytes, checkedMultiply((snapshotNodeCount_ + 63) / 64, sizeof(uint64_t), "preview mask"), "selector GPU");
		if(gpuBytes > config_.gpuBudgetBytes) return fail("selector total GPU buffer budget exceeded"), false;
	} catch(const std::overflow_error& e) {
		return fail(e.what()), false;
	}
	ensureDevice(dStageRows_, stageRowsCapacity_, std::size_t(potentialVisibleSplats) * sizeof(uint32_t), "selector stage rows");
	ensureDevice(dLocalToSnapshot_, localToSnapshotCapacity_, localToSnapshot_.size() * sizeof(uint32_t), "selector local-to-snapshot map");
	ensureDevice(dInitialDepth_, depthCapacity_, pixels * sizeof(float), "selector initial depth");
	ensureDevice(dForegroundMask_, foregroundMaskCapacity_, pixels, "selector FG mask");
	ensureDevice(dBackgroundMask_, backgroundMaskCapacity_, pixels, "selector BG mask");
	ensureDevice(dAllMass_, allMassCapacity_, n * sizeof(float), "selector all mass");
	ensureDevice(dForegroundMass_, foregroundMassCapacity_, n * sizeof(float), "selector FG mass");
	ensureDevice(dBackgroundMass_, backgroundMassCapacity_, n * sizeof(float), "selector BG mass");
	ensureDevice(dSupportCenter_, supportCenterCapacity_, n * 4 * sizeof(float), "selector support/center");
	CURuntime::check(cuMemcpyHtoD(dForegroundMask_, requestForegroundMask_.data(), pixels));
	CURuntime::check(cuMemcpyHtoD(dBackgroundMask_, requestBackgroundMask_.data(), pixels));
	CURuntime::check(cuMemcpyHtoD(dLocalToSnapshot_, localToSnapshot_.data(), localToSnapshot_.size() * sizeof(uint32_t)));
	cuMemsetD32(dAllMass_, 0, static_cast<unsigned int>(n));
	cuMemsetD32(dForegroundMass_, 0, static_cast<unsigned int>(n));
	cuMemsetD32(dBackgroundMass_, 0, static_cast<unsigned int>(n));
	cuMemsetD32(dSupportCenter_, 0, static_cast<unsigned int>(n*4));
	const std::size_t pinnedBytes = n * 7 * sizeof(float);
	if(pinnedBytes > hMassCapacity_) {
		if(hMassPinned_) cuMemFreeHost(hMassPinned_);
		CURuntime::check(cuMemAllocHost(reinterpret_cast<void**>(&hMassPinned_), pinnedBytes));
		hMassCapacity_ = pinnedBytes;
	}
	target.selectorStageRows = reinterpret_cast<uint32_t*>(dStageRows_);
	coveragePrepared_ = true;
	state_ = GaussianSelectorState::Lifting;
	message_ = "Replaying alpha-transmittance evidence";
	return true;
}

void GaussianSelectorRuntime::captureOpaqueDepth(RenderTarget target, CUstream stream) {
	if(!coveragePrepared_) return;
	editor_->prog_gaussians_rendering->launch("kernel_gaussian_selector_capture_depth",
		{&target, &dInitialDepth_}, target.width * target.height, stream);
}

void GaussianSelectorRuntime::configureGaussianData(SNSplats* node, GaussianData& data) const {
	data.selectorTarget = target_ && node == target_.get();
	data.selectorSnapshotBase = 0;
	data.selectorLocalToSnapshot = data.selectorTarget ? reinterpret_cast<const uint32_t*>(dLocalToSnapshot_) : nullptr;
	data.selectorPreviewWords = data.selectorTarget && hasPreview()
		? reinterpret_cast<const uint64_t*>(dPreviewWords_) : nullptr;
}

void GaussianSelectorRuntime::launchCoverage(RenderTarget target, CUdeviceptr tiles,
	CUdeviceptr tileIds, CUdeviceptr indices, CUdeviceptr stageData, uint32_t numTiles,
	uint32_t numFragments, CUstream stream) {
	if(!coveragePrepared_ || coverageInFlight_) return;
	const std::size_t measurementBytes=snapshotRows_->size()*sizeof(float);
	CUdeviceptr fgSupport=dSupportCenter_,bgSupport=dSupportCenter_+measurementBytes,
		fgCenter=dSupportCenter_+2*measurementBytes,bgCenter=dSupportCenter_+3*measurementBytes;
	void* args[] = {&editor_->launchArgs, &target, &tiles, &tileIds, &indices,
		&stageData, &numFragments, &dStageRows_,
		&dForegroundMask_, &dBackgroundMask_, &dInitialDepth_,
		&dAllMass_, &dForegroundMass_, &dBackgroundMass_, &requestEvidenceStop_, &requestOwnerBudget_,
		&fgSupport,&bgSupport,&fgCenter,&bgCenter,&requestCenterTolerance_};
	editor_->prog_gaussians_rendering->launch("kernel_gaussian_selector_coverage", args,
		{.gridsize = numTiles, .blocksize = 256, .stream = stream});
	const std::size_t nBytes = snapshotRows_->size() * sizeof(float);
	CURuntime::check(cuMemcpyDtoHAsync(hMassPinned_, dAllMass_, nBytes, stream));
	CURuntime::check(cuMemcpyDtoHAsync(hMassPinned_ + snapshotRows_->size(), dForegroundMass_, nBytes, stream));
	CURuntime::check(cuMemcpyDtoHAsync(hMassPinned_ + 2 * snapshotRows_->size(), dBackgroundMass_, nBytes, stream));
	CURuntime::check(cuMemcpyDtoHAsync(hMassPinned_ + 3 * snapshotRows_->size(), dSupportCenter_, nBytes*4, stream));
	cuEventRecord(coverageDone_, stream);
	coverageRequested_ = false;
	coveragePrepared_ = false;
	coverageInFlight_ = true;
}

void GaussianSelectorRuntime::finishEmptyCoverage(CUstream stream) {
	if(!coveragePrepared_ || coverageInFlight_) return;
	const std::size_t nBytes = snapshotRows_->size() * sizeof(float);
	CURuntime::check(cuMemcpyDtoHAsync(hMassPinned_, dAllMass_, nBytes, stream));
	CURuntime::check(cuMemcpyDtoHAsync(hMassPinned_ + snapshotRows_->size(), dForegroundMass_, nBytes, stream));
	CURuntime::check(cuMemcpyDtoHAsync(hMassPinned_ + 2 * snapshotRows_->size(), dBackgroundMass_, nBytes, stream));
	CURuntime::check(cuMemcpyDtoHAsync(hMassPinned_ + 3 * snapshotRows_->size(), dSupportCenter_, nBytes*4, stream));
	cuEventRecord(coverageDone_, stream);
	coverageRequested_ = false;
	coveragePrepared_ = false;
	coverageInFlight_ = true;
}

void GaussianSelectorRuntime::consumeCoverage() {
	if(!snapshotStillValid()) {
		state_ = GaussianSelectorState::Stale;
		message_ = "Gaussian indices changed while lifting evidence";
		return;
	}
	const std::size_t n = snapshotRows_->size();
	std::vector<CoverageMass> mass(n);
	for(std::size_t i = 0; i < n; ++i) {
		mass[i] = {hMassPinned_[i], hMassPinned_[n + i], hMassPinned_[2 * n + i],
			hMassPinned_[3*n+i],hMassPinned_[4*n+i],hMassPinned_[5*n+i],hMassPinned_[6*n+i]};
	}
	viewEvidence_[pendingViewKey_] = classifyEvidence(mass, config_);
	coverageReady_ = true;
	if(!updateRoiFromEvidence()) return;
	tryStartSolve();
}

bool GaussianSelectorRuntime::updateRoiFromEvidence() {
	if(screenRoi_) {
		// Snapshot is already compacted to the rectangle. Reuse its row order;
		// neither a second projection nor a second SH snapshot copy is needed.
		roiActive_ = true;
		encodingRowToSnapshot_.clear();
		if(!encoding_) startEncoding(requestId_);
		else encodingReady_=true;
		return true;
	}
	if(snapshotRows_->size() <= config_.fullScopeWarningCount) {
		roiActive_ = false;
		encodingRowToSnapshot_.clear();
		return true;
	}
	std::vector<std::vector<SeedLabel>> views;
	views.reserve(viewEvidence_.size());
	for(const auto& [_, evidence] : viewEvidence_) views.push_back(evidence.labels);
	const auto voted = majorityVote(views);
	Vec3 lo{std::numeric_limits<float>::infinity(),std::numeric_limits<float>::infinity(),std::numeric_limits<float>::infinity()};
	Vec3 hi{-lo.x,-lo.y,-lo.z},sceneLo=lo,sceneHi=hi;
	std::size_t fgCount = 0;
	for(std::size_t i=0;i<voted.size();++i){const auto p=(*snapshotRows_)[i].position;sceneLo.x=std::min(sceneLo.x,p.x);sceneLo.y=std::min(sceneLo.y,p.y);sceneLo.z=std::min(sceneLo.z,p.z);sceneHi.x=std::max(sceneHi.x,p.x);sceneHi.y=std::max(sceneHi.y,p.y);sceneHi.z=std::max(sceneHi.z,p.z);if(voted[i]==SeedLabel::Foreground){lo.x=std::min(lo.x,p.x);lo.y=std::min(lo.y,p.y);lo.z=std::min(lo.z,p.z);hi.x=std::max(hi.x,p.x);hi.y=std::max(hi.y,p.y);hi.z=std::max(hi.z,p.z);++fgCount;}}
	if(!fgCount) { fail("no high-confidence foreground Gaussian was lifted; enlarge the FG stroke or add another view"); return false; }
	const float dx=hi.x-lo.x,dy=hi.y-lo.y,dz=hi.z-lo.z;
	const float diagonal=std::max(std::sqrt(dx*dx+dy*dy+dz*dz),1e-6f),sdx=sceneHi.x-sceneLo.x,sdy=sceneHi.y-sceneLo.y,sdz=sceneHi.z-sceneLo.z,sceneDiagonal=std::sqrt(sdx*sdx+sdy*sdy+sdz*sdz);
	const float padding=std::max(config_.roiForegroundPaddingFraction*diagonal,
		config_.roiScenePaddingFraction*sceneDiagonal);
	lo.x-=padding;lo.y-=padding;lo.z-=padding;hi.x+=padding;hi.y+=padding;hi.z+=padding;
	std::vector<uint32_t> next;
	next.reserve(std::min(snapshotRows_->size(),config_.fullScopeWarningCount));
	for(std::size_t i=0;i<snapshotRows_->size();++i){const auto p=(*snapshotRows_)[i].position;const bool inside=p.x>=lo.x&&p.x<=hi.x&&p.y>=lo.y&&p.y<=hi.y&&p.z>=lo.z&&p.z<=hi.z;if(inside||voted[i]!=SeedLabel::Unknown)next.push_back(static_cast<uint32_t>(i));}
	std::size_t estimate = config_.cpuBudgetBytes;
	try {
		const auto edgeBytes = checkedMultiply(std::size_t(config_.knn), 2 * sizeof(WeightedEdge), "ROI edges");
		const auto bytesPerRow = checkedAdd(sizeof(GaussianSnapshotRow) + kCacDimensions * sizeof(float), edgeBytes, "ROI row");
		estimate = checkedMultiply(next.size(), bytesPerRow, "ROI encoding");
	} catch(const std::overflow_error&) {}
	if(estimate>=config_.cpuBudgetBytes){fail("automatic ROI exceeds CPU budget; use a tighter FG stroke, reduce ROI padding, or isolate the layer");return false;}
	if(next.empty()){fail("automatic ROI is empty");return false;}
	roiActive_=true;
	if(next!=encodingRowToSnapshot_||!encoding_){encodingRowToSnapshot_=std::move(next);encoding_.reset();encodingReady_=false;++roiRevision_;startEncoding(requestId_);message_="Encoding expanded 3D ROI from lifted FG seeds (ROI outside still occludes coverage)";}
	else encodingReady_=true;
	return true;
}

void GaussianSelectorRuntime::tryStartSolve() {
	if(!encodingReady_ || !coverageReady_ || solveInFlight_ || !encoding_) return;
	std::vector<std::vector<SeedLabel>> views;
	views.reserve(viewEvidence_.size());
	std::vector<std::vector<SeedConfidence>> confidenceViews;
	for(const auto& [_, evidence] : viewEvidence_) {views.push_back(evidence.labels);confidenceViews.push_back(evidence.confidence);}
	auto seeds = majorityVote(views);
	auto confidence = averageConfidence(confidenceViews);
	if(!encodingRowToSnapshot_.empty()) { std::vector<SeedLabel> scoped; scoped.reserve(encodingRowToSnapshot_.size()); for(auto row:encodingRowToSnapshot_) scoped.push_back(seeds[row]); seeds=std::move(scoped); }
	if(!encodingRowToSnapshot_.empty()) {std::vector<SeedConfidence> scoped;scoped.reserve(encodingRowToSnapshot_.size());for(auto row:encodingRowToSnapshot_)scoped.push_back(confidence[row]);confidence=std::move(scoped);}
	auto encoding = encoding_;
	auto config = config_;
	auto key = JobKey{snapshotKey_, static_cast<uint64_t>(viewEvidence_.size()), roiRevision_, requestId_};
	auto mailbox = mailbox_;
	const uint64_t id = requestId_;
	solveInFlight_ = true;
	state_ = GaussianSelectorState::Solving;
	message_ = "Fitting FG/BG GMM and solving s-t min-cut";
	std::thread([encoding, config, seeds = std::move(seeds), confidence = std::move(confidence), key, mailbox, id] {
		SolvePublication publication;
		publication.requestId = id;
		try {
			SelectorSession session(config);
			session.setEncoding(encoding);
			publication.result = session.solve(key, seeds, confidence);
		} catch(const std::exception& e) {
			publication.error = e.what();
		}
		std::scoped_lock lock(mailbox->mutex);
		mailbox->solves.push_back(std::move(publication));
	}).detach();
}

void GaussianSelectorRuntime::publishPreview(const SelectionResult& result) {
	if(!snapshotStillValid() || result.key.snapshot != snapshotKey_ || visibilityRevision()!=evidenceVisibilityRevision_) {
		state_ = GaussianSelectorState::Stale;
		message_ = "discarded stale GaussianSelector result";
		return;
	}
	previewWords_ = result.maskWords;
	previewLocalWords_.assign((snapshotNodeCount_ + 63) / 64, 0);
	for(std::size_t row = 0; row < (encodingRowToSnapshot_.empty()?snapshotRows_->size():encodingRowToSnapshot_.size()); ++row) if(maskGet(previewWords_, row)){const auto snapshotRow=encodingRowToSnapshot_.empty()?row:std::size_t(encodingRowToSnapshot_[row]);maskSet(previewLocalWords_, (*snapshotRows_)[snapshotRow].identity.localIndex, true);}
	diagnostics_ = result.diagnostics;
	ensureDevice(dPreviewWords_, previewCapacity_, previewLocalWords_.size() * sizeof(uint64_t), "selector preview words");
	if(!previewLocalWords_.empty()) CURuntime::check(cuMemcpyHtoD(dPreviewWords_, previewLocalWords_.data(), previewLocalWords_.size() * sizeof(uint64_t)));
	state_ = GaussianSelectorState::PreviewReady;
	message_ = "Preview ready; FLAGS_SELECTED is unchanged";
}

void GaussianSelectorRuntime::poll() {
	if(coverageInFlight_) {
		const CUresult status = cuEventQuery(coverageDone_);
		if(status == CUDA_SUCCESS) {
			coverageInFlight_ = false;
			consumeCoverage();
		} else if(status != CUDA_ERROR_NOT_READY) {
			fail("coverage CUDA event failed");
		}
	}

	std::vector<EncodingPublication> encodingPublications;
	std::vector<SolvePublication> solvePublications;
	{
		std::scoped_lock lock(mailbox_->mutex);
		encodingPublications.swap(mailbox_->encodings);
		solvePublications.swap(mailbox_->solves);
	}
	for(auto& publication : encodingPublications) {
		if(publication.requestId != requestId_) continue;
		if(!publication.error.empty()) return fail(publication.error);
		encoding_ = std::move(publication.encoding);
		encodingReady_ = true;
		tryStartSolve();
	}
	for(auto& publication : solvePublications) {
		if(publication.requestId != requestId_) continue;
		solveInFlight_ = false;
		if(!publication.error.empty()) return fail(publication.error);
		publishPreview(*publication.result);
	}
}

void GaussianSelectorRuntime::cancel() {
	requestId_++;
	coverageRequested_ = coveragePrepared_ = false;
	coverageInFlight_ = false;
	solveInFlight_ = false;
	state_ = GaussianSelectorState::Cancelled;
	message_ = "Cancelled; late worker results will be discarded";
}

void GaussianSelectorRuntime::resetPreview() {
	previewWords_.clear();
	previewLocalWords_.clear();
	if(state_ == GaussianSelectorState::PreviewReady) state_ = GaussianSelectorState::Idle;
}

void GaussianSelectorRuntime::clearEvidence(uint64_t key) {
	requestId_++;
	coverageRequested_ = coveragePrepared_ = coverageInFlight_ = solveInFlight_ = false;
	viewEvidence_.erase(key);
	resetPreview();
	state_ = GaussianSelectorState::Idle;
	message_ = "Current-view evidence cleared; outstanding results invalidated";
}

void GaussianSelectorRuntime::clearAllEvidence() {
	requestId_++;
	coverageRequested_ = coveragePrepared_ = coverageInFlight_ = solveInFlight_ = false;
	viewEvidence_.clear();
	resetPreview();
	state_ = GaussianSelectorState::Idle;
	message_ = "All evidence cleared; outstanding results invalidated";
}

bool GaussianSelectorRuntime::apply(ApplyMode mode) {
	if(!hasPreview() || !snapshotStillValid() || !target_ || target_->locked || visibilityRevision()!=evidenceVisibilityRevision_) {
		message_ = "Apply refused: preview or target epoch is stale";
		return false;
	}
	const uint32_t n = target_->dmng.data.count;
	std::vector<uint32_t> flags(n);
	CURuntime::check(cuMemcpyDtoH(flags.data(), reinterpret_cast<CUdeviceptr>(target_->dmng.data.flags), n * sizeof(uint32_t)));
	const std::size_t words = (n + 31u) / 32u;
	std::vector<uint32_t> before(words), after(words);
	std::vector<uint8_t> domain(n, 0);
	if(encodingRowToSnapshot_.empty()) {
		for(const auto& row : *snapshotRows_) domain[row.identity.localIndex]=1;
	} else {
		for(auto row : encodingRowToSnapshot_) domain[(*snapshotRows_)[row].identity.localIndex]=1;
	}
	bool changed = false;
	for(uint32_t i = 0; i < n; ++i) {
		const bool was = (flags[i] & FLAGS_SELECTED) != 0;
		const bool preview = maskGet(previewLocalWords_, i);
		bool next = was;
		if(domain[i] && (flags[i] & FLAGS_DELETED) == 0) {
			switch(mode) {
				case ApplyMode::Replace: next = preview; break;
				case ApplyMode::Add: next = was || preview; break;
				case ApplyMode::Subtract: next = was && !preview; break;
				case ApplyMode::Intersect: next = was && preview; break;
			}
		}
		if(was) before[i >> 5] |= 1u << (i & 31u);
		if(next) after[i >> 5] |= 1u << (i & 31u);
		changed |= was != next;
	}
	if(!changed) {
		message_ = "Apply produced no selection change";
		return false;
	}
	auto undo = std::make_shared<GaussianSelectorSelectionUndoAction>();
	undo->node = target_;
	undo->indexEpoch = target_->indexEpoch;
	undo->count = n;
	undo->swapMask = CURuntime::alloc("GaussianSelector selection before", words * sizeof(uint32_t));
	CURuntime::check(cuMemcpyHtoD(undo->swapMask, before.data(), words * sizeof(uint32_t)));
	CUdeviceptr afterMask = CURuntime::alloc("GaussianSelector selection after", words * sizeof(uint32_t));
	CURuntime::check(cuMemcpyHtoD(afterMask, after.data(), words * sizeof(uint32_t)));
	bool inverted = false;
	editor_->prog_gaussians_editing->launch("kernel_set_selection",
		{&target_->dmng.data, &afterMask, &inverted}, n);
	CURuntime::free(afterMask);
	editor_->addAction(undo);
	message_ = "Selection applied as one exact Undo/Redo command";
	return true;
}

#endif
