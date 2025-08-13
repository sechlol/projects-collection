#pragma once
#include "SuffixMapData.h"
#include <unordered_map>

namespace LCS{

class SlidingWindow {
private:
    size_t currentSize;
    SuffixMapData::iterator begin;
    SuffixMapData::iterator end;
    SuffixMapData::iterator top;
    SuffixMapData::iterator bottom;
    std::unordered_map<int, int> sourceCount;
    bool IncreaseFromTop();
public:
    SlidingWindow(SuffixMapData::iterator begin, SuffixMapData::iterator end);
    [[nodiscard]] size_t Size() const;
    [[nodiscard]] const SuffixMapData::iterator& End() const;
    [[nodiscard]] const SuffixMapData::iterator& Top() const;
    [[nodiscard]] const SuffixMapData::iterator& Bottom() const;

    bool IncreaseFromBottom();
    bool DecreaseFromTop();
    bool FitTopToContain(int desiredSources);

    [[nodiscard]] int QueryNumberOfDifferentSources() const;
    [[nodiscard]] SuffixMapData::iterator QueryLCSInRange() const;
};

}