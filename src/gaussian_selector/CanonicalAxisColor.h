#pragma once

#include "SelectorTypes.h"

namespace gaussian_selector {

Vec3 evaluateHostAlignedSh(const GaussianSnapshotRow& row, Vec3 direction);
CacFeature computeCanonicalAxisColor(const GaussianSnapshotRow& row);

} // namespace gaussian_selector
