#include "SelectorSelectionCommand.h"

#include <stdexcept>

namespace gaussian_selector {

SelectionDelta makeSelectionDelta(SnapshotKey key,std::span<const std::uint32_t> flags,std::span<const std::uint64_t> preview,std::span<const std::uint8_t> domain,ApplyMode mode,std::uint32_t bit){if(flags.size()!=domain.size())throw std::invalid_argument("selection domain mismatch");SelectionDelta d;d.key=key;for(std::size_t i=0;i<flags.size();i++){if(!domain[i])continue;bool before=(flags[i]&bit)!=0,p=i/64<preview.size()&&((preview[i/64]>>(i%64))&1u),after=before;switch(mode){case ApplyMode::Replace:after=p;break;case ApplyMode::Add:after=before||p;break;case ApplyMode::Subtract:after=before&&!p;break;case ApplyMode::Intersect:after=before&&p;break;}if(before!=after){d.indices.push_back(static_cast<std::uint32_t>(i));d.before.push_back(before);d.after.push_back(after);}}return d;}

void applySelectionDelta(std::span<std::uint32_t> flags,const SelectionDelta& d,bool redo,std::uint32_t bit){for(std::size_t j=0;j<d.indices.size();j++){auto i=d.indices[j];if(i>=flags.size())throw std::out_of_range("selection delta stale");const bool v=redo?d.after[j]:d.before[j];flags[i]=(flags[i]&~bit)|(v?bit:0);}}

} // namespace gaussian_selector
