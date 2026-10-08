#pragma once

#ifdef SPLATSHOP_ENABLE_GAUSSIAN_SELECTOR

#include "gaussian_selector/SelectorTypes.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
#include <functional>
#include <glm/glm.hpp>

struct SNSplats;

struct SceneSnapshotCapture {
	std::shared_ptr<std::vector<gaussian_selector::GaussianSnapshotRow>> rows;
	std::vector<std::uint32_t> localToSnapshot;
	std::size_t nodeCount = 0;
	std::size_t skippedDeleted = 0;
	std::size_t skippedNonFinite = 0;
	std::uint64_t transformRevision = 0;
};

class SceneSnapshotAdapter {
public:
	// Filter world-space centers before reading SH/scale/color. Empty filter = full layer.
	static SceneSnapshotCapture capture(const SNSplats& node,
		const std::function<bool(glm::vec3)>& include,
		std::size_t cpuBudgetBytes, std::size_t maxRows, std::size_t graphBytesPerRow);
	static std::uint64_t transformRevision(const SNSplats& node);
};

#endif
