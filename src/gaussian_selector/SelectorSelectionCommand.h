#pragma once

#include "SelectorTypes.h"

#include <span>

namespace gaussian_selector {

enum class ApplyMode { Replace, Add, Subtract, Intersect };

struct SelectionDelta { SnapshotKey key; std::vector<std::uint32_t> indices; std::vector<std::uint8_t> before,after; bool empty()const{return indices.empty();} };
SelectionDelta makeSelectionDelta(SnapshotKey key,std::span<const std::uint32_t> flags,
    std::span<const std::uint64_t> preview,std::span<const std::uint8_t> inDomain,
    ApplyMode mode,std::uint32_t selectedBit);
void applySelectionDelta(std::span<std::uint32_t> flags,const SelectionDelta& delta,bool redo,std::uint32_t selectedBit);

} // namespace gaussian_selector
