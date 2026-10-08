#pragma once

#ifdef SPLATSHOP_ENABLE_GAUSSIAN_SELECTOR

#include <cstdint>

#include <cuda.h>
#include <glm/glm.hpp>

#include "gaussian_selector/SelectorConfig.h"
#include "gaussian_selector/SelectorSession.h"
#include "gaussian_selector/SelectorSelectionCommand.h"

#include <atomic>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

struct SplatEditor;
struct SNSplats;
struct RenderTarget;
struct GaussianData;

using glm::mat4;

enum class GaussianSelectorState {
	Idle,
	Snapshotting,
	Encoding,
	Lifting,
	Solving,
	PreviewReady,
	Cancelled,
	Failed,
	Stale,
};

class GaussianSelectorRuntime {
public:
	explicit GaussianSelectorRuntime(SplatEditor* editor);
	~GaussianSelectorRuntime();

	static uint64_t viewKey(int width, int height, const mat4& view, const mat4& proj);
	uint64_t evidenceViewKey(int width, int height, const mat4& view, const mat4& proj) const;
	void requestSolve(const std::shared_ptr<SNSplats>& target,
		int width, int height, const mat4& view, const mat4& proj,
		const std::vector<uint8_t>& foregroundMask,
		const std::vector<uint8_t>& backgroundMask,
		std::optional<glm::vec4> screenRoi = std::nullopt);
	void poll();
	void cancel();
	void resetPreview();
	void clearEvidence(uint64_t key);
	void clearAllEvidence();
	bool apply(gaussian_selector::ApplyMode mode);

	bool prepareStandardTarget(RenderTarget& target, uint32_t potentialVisibleSplats);
	void captureOpaqueDepth(RenderTarget target, CUstream stream);
	void configureGaussianData(SNSplats* node, GaussianData& data) const;
	void launchCoverage(RenderTarget target, CUdeviceptr tiles, CUdeviceptr tileIds,
		CUdeviceptr indices, CUdeviceptr stageData, uint32_t numTiles,
		uint32_t numFragments, CUstream stream);
	void finishEmptyCoverage(CUstream stream);

	GaussianSelectorState state() const { return state_; }
	const std::string& message() const { return message_; }
	const gaussian_selector::SolveDiagnostics& diagnostics() const { return diagnostics_; }
	std::size_t gaussianCount() const { return snapshotRows_ ? snapshotNodeCount_ : 0; }
	std::size_t scopedGaussianCount() const { return encodingRowToSnapshot_.empty() ? (snapshotRows_ ? snapshotRows_->size() : 0) : encodingRowToSnapshot_.size(); }
	std::size_t superpointCount() const { return hasPreview() ? diagnostics_.solvedSuperpoints : (encoding_ ? encoding_->superpoints.size() : 0); }
	std::size_t edgeCount() const { return hasPreview() ? diagnostics_.solvedEdges : (encoding_ ? encoding_->superpointGraph.edges.size() : 0); }
	std::size_t evidenceViewCount() const { return viewEvidence_.size(); }
	std::size_t encodingNodeCount() const { return encoding_ ? encoding_->encodingNodeCount : 0; }
	bool voxelPooled() const { return encoding_ && encoding_->voxelPooled; }
	bool hasPreview() const { return state_ == GaussianSelectorState::PreviewReady && !previewWords_.empty(); }
	bool roiActive() const { return roiActive_; }
	uint64_t requestId() const { return requestId_; }
	gaussian_selector::SelectorConfig& config() { return config_; }

private:
	struct EncodingPublication {
		uint64_t requestId = 0;
		std::shared_ptr<gaussian_selector::SceneEncoding> encoding;
		std::string error;
	};
	struct SolvePublication {
		uint64_t requestId = 0;
		std::optional<gaussian_selector::SelectionResult> result;
		std::string error;
	};
	struct Mailbox {
		std::mutex mutex;
		std::vector<EncodingPublication> encodings;
		std::vector<SolvePublication> solves;
	};

	void captureSnapshot(const std::shared_ptr<SNSplats>& target);
	uint64_t visibilityRevision() const;
	void startEncoding(uint64_t requestId);
	void consumeCoverage();
	bool updateRoiFromEvidence();
	void tryStartSolve();
	void publishPreview(const gaussian_selector::SelectionResult& result);
	bool snapshotStillValid() const;
	void fail(std::string message);
	void ensureDevice(CUdeviceptr& pointer, std::size_t& capacity,
		std::size_t requiredBytes, const char* label);
	void releaseDeviceResources();

	SplatEditor* editor_ = nullptr;
	gaussian_selector::SelectorConfig config_;
	GaussianSelectorState state_ = GaussianSelectorState::Idle;
	std::string message_ = "Idle";
	gaussian_selector::SolveDiagnostics diagnostics_;
	std::shared_ptr<SNSplats> target_;
	gaussian_selector::SnapshotKey snapshotKey_;
	std::shared_ptr<std::vector<gaussian_selector::GaussianSnapshotRow>> snapshotRows_;
	std::vector<uint32_t> localToSnapshot_;
	std::size_t snapshotNodeCount_ = 0;
	uint64_t targetContentRevision_ = 0;
	uint64_t targetTransformRevision_ = 0;
	std::shared_ptr<gaussian_selector::SceneEncoding> encoding_;
	std::vector<uint32_t> encodingRowToSnapshot_;
	std::shared_ptr<Mailbox> mailbox_ = std::make_shared<Mailbox>();
	std::unordered_map<uint64_t, gaussian_selector::ViewEvidence> viewEvidence_;
	std::vector<uint64_t> previewWords_;
	std::vector<uint64_t> previewLocalWords_;

	uint64_t requestId_ = 0;
	uint64_t roiRevision_ = 0;
	bool roiActive_ = false;
	std::optional<glm::vec4> screenRoi_;
	uint64_t pendingViewKey_ = 0;
	uint64_t evidenceVisibilityRevision_ = 0;
	uint64_t evidenceConfigRevision_ = 0;
	float requestEvidenceStop_ = 0;
	uint32_t requestOwnerBudget_ = 16;
	float requestCenterTolerance_ = .02f;
	int requestWidth_ = 0;
	int requestHeight_ = 0;
	mat4 requestView_{1.0f};
	mat4 requestProj_{1.0f};
	std::vector<uint8_t> requestForegroundMask_;
	std::vector<uint8_t> requestBackgroundMask_;
	bool coverageRequested_ = false;
	bool coveragePrepared_ = false;
	bool coverageInFlight_ = false;
	bool encodingReady_ = false;
	bool coverageReady_ = false;
	bool solveInFlight_ = false;

	CUdeviceptr dStageRows_ = 0;
	CUdeviceptr dLocalToSnapshot_ = 0;
	CUdeviceptr dInitialDepth_ = 0;
	CUdeviceptr dForegroundMask_ = 0;
	CUdeviceptr dBackgroundMask_ = 0;
	CUdeviceptr dAllMass_ = 0;
	CUdeviceptr dForegroundMass_ = 0;
	CUdeviceptr dBackgroundMass_ = 0;
	CUdeviceptr dSupportCenter_ = 0;
	CUdeviceptr dPreviewWords_ = 0;
	std::size_t stageRowsCapacity_ = 0;
	std::size_t localToSnapshotCapacity_ = 0;
	std::size_t depthCapacity_ = 0;
	std::size_t foregroundMaskCapacity_ = 0;
	std::size_t backgroundMaskCapacity_ = 0;
	std::size_t allMassCapacity_ = 0;
	std::size_t foregroundMassCapacity_ = 0;
	std::size_t backgroundMassCapacity_ = 0;
	std::size_t supportCenterCapacity_ = 0;
	std::size_t previewCapacity_ = 0;
	float* hMassPinned_ = nullptr;
	std::size_t hMassCapacity_ = 0;
	CUevent coverageDone_ = nullptr;
};

const char* gaussianSelectorStateName(GaussianSelectorState state);

#endif
