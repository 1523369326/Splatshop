#include "CanonicalAxisColor.h"

#include <algorithm>
#include <cmath>

namespace gaussian_selector {
namespace {

Vec3 normalize(Vec3 v) {
    const float n = std::sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
    if (!(n > 1e-20f) || !std::isfinite(n)) return {1, 0, 0};
    return {v.x/n, v.y/n, v.z/n};
}

Quaternion normalize(Quaternion q) {
    const float n = std::sqrt(q.w*q.w + q.x*q.x + q.y*q.y + q.z*q.z);
    if (!(n > 1e-20f) || !std::isfinite(n)) return {};
    return {q.w/n, q.x/n, q.y/n, q.z/n};
}

Vec3 rotate(Quaternion q, Vec3 v) {
    q = normalize(q);
    const Vec3 u{q.x, q.y, q.z};
    const float uv = u.x*v.x + u.y*v.y + u.z*v.z;
    const float uu = u.x*u.x + u.y*u.y + u.z*u.z;
    const Vec3 cross{u.y*v.z-u.z*v.y, u.z*v.x-u.x*v.z, u.x*v.y-u.y*v.x};
    return {2*uv*u.x + (q.w*q.w-uu)*v.x + 2*q.w*cross.x,
            2*uv*u.y + (q.w*q.w-uu)*v.y + 2*q.w*cross.y,
            2*uv*u.z + (q.w*q.w-uu)*v.z + 2*q.w*cross.z};
}

Vec3 coefficient(const GaussianSnapshotRow& row, std::size_t i) {
    const std::size_t base = i * 3;
    if (base + 2 >= row.shRest.size()) return {};
    return {row.shRest[base], row.shRest[base+1], row.shRest[base+2]};
}

Vec3 addScaled(Vec3 a, float s, Vec3 b) {
    return {a.x+s*b.x, a.y+s*b.y, a.z+s*b.z};
}

} // namespace

Vec3 evaluateHostAlignedSh(const GaussianSnapshotRow& row, Vec3 d) {
    d = normalize(d);
    Vec3 result{row.baseColor[0], row.baseColor[1], row.baseColor[2]};
    constexpr float C1 = 0.4886025119029199f;
    constexpr float C2[5] = {1.0925484305920792f, -1.0925484305920792f,
        0.31539156525252005f, -1.0925484305920792f, 0.5462742152960396f};
    constexpr float C3[7] = {-0.5900435899266435f, 2.890611442640554f,
        -0.4570457994644658f, 0.3731763325901154f, -0.4570457994644658f,
        1.445305721320277f, -0.5900435899266435f};
    if (row.shDegree > 0) {
        result = addScaled(result, -C1*d.y, coefficient(row, 0));
        result = addScaled(result,  C1*d.z, coefficient(row, 1));
        result = addScaled(result, -C1*d.x, coefficient(row, 2));
    }
    const float xx=d.x*d.x, yy=d.y*d.y, zz=d.z*d.z;
    const float xy=d.x*d.y, yz=d.y*d.z, xz=d.x*d.z;
    if (row.shDegree > 1) {
        const float b[5] = {C2[0]*xy, C2[1]*yz, C2[2]*(2*zz-xx-yy),
                            C2[3]*xz, C2[4]*(xx-yy)};
        for (std::size_t i=0;i<5;i++) result=addScaled(result,b[i],coefficient(row,i+3));
    }
    if (row.shDegree > 2) {
        const float b[7] = {C3[0]*d.y*(3*xx-yy), C3[1]*xy*d.z,
            C3[2]*d.y*(4*zz-xx-yy), C3[3]*d.z*(2*zz-3*xx-3*yy),
            C3[4]*d.x*(4*zz-xx-yy), C3[5]*d.z*(xx-yy),
            C3[6]*d.x*(xx-3*yy)};
        for (std::size_t i=0;i<7;i++) result=addScaled(result,b[i],coefficient(row,i+8));
    }
    // Match the host's post-SH lower clamp; the upper clamp occurs in its
    // 8-bit staging conversion.  CAC remains float and clamps to [0,1] as an
    // explicit engineering choice aligned with the author's implementation note.
    result.x=std::clamp(result.x,0.0f,1.0f);
    result.y=std::clamp(result.y,0.0f,1.0f);
    result.z=std::clamp(result.z,0.0f,1.0f);
    return result;
}

CacFeature computeCanonicalAxisColor(const GaussianSnapshotRow& row) {
    const Vec3 axes[6]={{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1}};
    CacFeature out{};
    for (std::size_t i=0;i<6;i++) {
        // Splatshop stores physical scale, not log-scale.  For canonical basis
        // axes the positive scale factor cancels during normalization.
        const Vec3 c=evaluateHostAlignedSh(row,rotate(row.rotation,axes[i]));
        out[i*3]=c.x; out[i*3+1]=c.y; out[i*3+2]=c.z;
    }
    return out;
}

} // namespace gaussian_selector
