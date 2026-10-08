#ifdef SPLATSHOP_ENABLE_GAUSSIAN_SELECTOR

#include "SceneSnapshotAdapter.h"

#include "SplatEditor.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <algorithm>
#include <stdexcept>

using namespace gaussian_selector;

namespace {

std::uint64_t hashBytes(std::uint64_t hash, const void* data, std::size_t bytes) {
	const auto* p = static_cast<const std::uint8_t*>(data);
	for(std::size_t i = 0; i < bytes; ++i) {
		hash ^= p[i];
		hash *= 1099511628211ull;
	}
	return hash;
}

bool finite(glm::vec3 v) { return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z); }
bool finite(glm::vec4 v) { return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z) && std::isfinite(v.w); }

} // namespace

std::uint64_t SceneSnapshotAdapter::transformRevision(const SNSplats& node) {
	return hashBytes(1469598103934665603ull, &node.transform_global, sizeof(node.transform_global));
}

SceneSnapshotCapture SceneSnapshotAdapter::capture(const SNSplats& node,
	const std::function<bool(glm::vec3)>& include, std::size_t cpuBudgetBytes,
	std::size_t maxRows, std::size_t graphBytesPerRow) {
	const auto& data = node.dmng.data;
	if(data.numSHCoefficients < 0 || data.numSHCoefficients > 45)
		throw std::runtime_error("unsupported SH coefficient count");
	const std::size_t shCount = std::size_t(data.numSHCoefficients);
	const std::size_t shBytes = shCount * sizeof(float);
	constexpr std::size_t chunkSize = 16384;
	SceneSnapshotCapture out;
	out.nodeCount = data.count;
	out.transformRevision = transformRevision(node);
	out.rows = std::make_shared<std::vector<GaussianSnapshotRow>>();
	// Account for the dense identity map, bounded transfer scratch, candidate
	// indices, snapshot SH allocations and an encoding working-set allowance.
	const std::size_t scratchBytes = chunkSize * (sizeof(glm::vec3)*2 +
		sizeof(glm::vec4)+sizeof(Color)+sizeof(uint32_t)+shBytes);
	if(out.nodeCount > cpuBudgetBytes / sizeof(uint32_t))
		throw std::runtime_error("selector identity map exceeds CPU budget");
	const std::size_t mapBytes = out.nodeCount * sizeof(uint32_t);
	if(scratchBytes > cpuBudgetBytes-mapBytes)
		throw std::runtime_error("selector transfer scratch exceeds CPU budget");
	const std::size_t available = cpuBudgetBytes-mapBytes-scratchBytes;
	const std::size_t bytesPerRow = 2*sizeof(GaussianSnapshotRow)+2*shBytes+
		2*sizeof(uint32_t)+64+graphBytesPerRow;
	const std::size_t rowLimit = std::min(maxRows,available/bytesPerRow);
	out.localToSnapshot.assign(out.nodeCount,kInvalidRow);
	std::vector<uint32_t> candidates;
	std::vector<glm::vec3> positions(chunkSize);
	std::vector<uint32_t> flags(chunkSize);
	// First pass has a fixed memory bound independent of the layer's SH size.
	for(std::size_t base=0;base<out.nodeCount;base+=chunkSize) {
		const std::size_t count=std::min(chunkSize,out.nodeCount-base);
		CURuntime::check(cuMemcpyDtoH(positions.data(),reinterpret_cast<CUdeviceptr>(data.position+base),count*sizeof(glm::vec3)));
		CURuntime::check(cuMemcpyDtoH(flags.data(),reinterpret_cast<CUdeviceptr>(data.flags+base),count*sizeof(uint32_t)));
		for(std::size_t i=0;i<count;++i) {
			if(flags[i]&FLAGS_DELETED) {++out.skippedDeleted;continue;}
			const auto world=node.transform_global*glm::vec4(positions[i],1);
			if(!finite(positions[i]) || !finite(world)) {++out.skippedNonFinite;continue;}
			if(include && !include(glm::vec3(world))) continue;
			if(candidates.size()>=rowLimit) {
				throw std::runtime_error("ROI candidate count exceeds "+std::to_string(rowLimit)+
					" (graph/CPU budget); draw a smaller ROI");
			}
			candidates.push_back(uint32_t(base+i));
		}
	}
	out.rows->reserve(candidates.size());
	std::vector<glm::vec3> scales(chunkSize);
	std::vector<glm::vec4> quaternions(chunkSize);
	std::vector<Color> colors(chunkSize);
	std::vector<float> sh(chunkSize*shCount);
	// Read only selected contiguous runs. Original local indices are retained;
	// gaps and outside-ROI entries remain invalid in the identity sidecar.
	for(std::size_t begin=0;begin<candidates.size();) {
		std::size_t end=begin+1;
		while(end<candidates.size() && end-begin<chunkSize &&
			candidates[end]==candidates[end-1]+1) ++end;
		const std::size_t base=candidates[begin],count=end-begin;
		CURuntime::check(cuMemcpyDtoH(positions.data(),reinterpret_cast<CUdeviceptr>(data.position+base),count*sizeof(glm::vec3)));
		CURuntime::check(cuMemcpyDtoH(scales.data(),reinterpret_cast<CUdeviceptr>(data.scale+base),count*sizeof(glm::vec3)));
		CURuntime::check(cuMemcpyDtoH(quaternions.data(),reinterpret_cast<CUdeviceptr>(data.quaternion+base),count*sizeof(glm::vec4)));
		CURuntime::check(cuMemcpyDtoH(colors.data(),reinterpret_cast<CUdeviceptr>(data.color+base),count*sizeof(Color)));
		if(shCount) CURuntime::check(cuMemcpyDtoH(sh.data(),reinterpret_cast<CUdeviceptr>(data.sphericalHarmonics+base*shCount),count*shBytes));
		for(std::size_t i=0;i<count;++i) {
			bool finiteSh=true;
			for(std::size_t c=0;c<shCount;++c) if(!std::isfinite(sh[i*shCount+c])) {finiteSh=false;break;}
			if(!finite(scales[i]) || !finite(quaternions[i]) || !finiteSh) {++out.skippedNonFinite;continue;}
			const auto world=node.transform_global*glm::vec4(positions[i],1);
			GaussianSnapshotRow row;
			row.identity={uint64_t(node.ID),node.indexEpoch,uint32_t(base+i)};
			row.position={world.x,world.y,world.z};
			row.physicalScale={scales[i].x,scales[i].y,scales[i].z};
			// Match renderer glm::quat(stored.x,stored.y,stored.z,stored.w).
			row.rotation={quaternions[i].x,quaternions[i].y,quaternions[i].z,quaternions[i].w};
			row.baseColor={Color::normalize(colors[i].r),Color::normalize(colors[i].g),Color::normalize(colors[i].b)};
			row.opacity=Color::normalize(colors[i].a);
			row.shDegree=data.shDegree;
			row.shRest.assign(sh.begin()+i*shCount,sh.begin()+(i+1)*shCount);
			out.localToSnapshot[base+i]=uint32_t(out.rows->size());
			out.rows->push_back(std::move(row));
		}
		begin=end;
	}
	return out;
}

#endif
