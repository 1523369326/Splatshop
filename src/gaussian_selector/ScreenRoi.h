#pragma once
#include <algorithm>
#include <cmath>
namespace gaussian_selector {
// Renderer screen coordinates have their origin at bottom left.
struct ScreenRoi {
    float x0, y0, x1, y1;
    bool contains(float x, float y) const {
        return std::isfinite(x) && std::isfinite(y) &&
            x>=std::min(x0,x1) && x<=std::max(x0,x1) &&
            y>=std::min(y0,y1) && y<=std::max(y0,y1);
    }
    bool containsClip(float x,float y,float z,float w,int width,int height) const {
        if(width<=0 || height<=0 || !std::isfinite(z) || !std::isfinite(w) || w<=0 || z < -w || z > w) return false;
        return contains((x/w*.5f+.5f)*width,(y/w*.5f+.5f)*height);
    }
};
}
