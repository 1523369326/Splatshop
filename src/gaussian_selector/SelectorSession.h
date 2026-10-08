#pragma once

#include "SelectorConfig.h"
#include "SelectorTypes.h"

#include <span>
#include <memory>

namespace gaussian_selector {

class SelectorSession {
public:
    explicit SelectorSession(SelectorConfig config={}):config_(config){}
    void setEncoding(SceneEncoding encoding){encoding_=std::make_shared<SceneEncoding>(std::move(encoding));}
    void setEncoding(std::shared_ptr<const SceneEncoding> encoding){encoding_=std::move(encoding);}
    const SceneEncoding& encoding()const{return *encoding_;}
    SelectionResult solve(JobKey key,std::span<const SeedLabel> gaussianSeeds,
        std::span<const SeedConfidence> confidence={})const;
private:
    SelectorConfig config_;
    std::shared_ptr<const SceneEncoding> encoding_;
};

} // namespace gaussian_selector
