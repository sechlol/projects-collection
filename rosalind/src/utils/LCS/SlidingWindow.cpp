#include "SlidingWindow.h"

namespace LCS{

    SlidingWindow::SlidingWindow(SuffixMapData::iterator iBegin, SuffixMapData::iterator iEnd) :
        currentSize(0),
        begin(iBegin),
        end(iEnd),
        top(iBegin),
        bottom(iBegin)
        {}

    bool SlidingWindow::IncreaseFromTop() {
       if (top == begin)
           return false;
       --top;
       ++sourceCount[top->second.ParentIndex];
       ++currentSize;
       return true;
    }

    bool SlidingWindow::IncreaseFromBottom() {
        if (bottom == end)
            return false;
        ++sourceCount[bottom->second.ParentIndex];
        ++currentSize;
        ++bottom;
        return true;
    }

    bool SlidingWindow::DecreaseFromTop() {
        if (top == bottom)
            return false;
        --sourceCount[top->second.ParentIndex];
        --currentSize;
        ++top++;
        return true;
    }

    size_t SlidingWindow::Size() const {
        return currentSize;
    }

    const SuffixMapData::iterator &SlidingWindow::End() const {
        return end;
    }

    const SuffixMapData::iterator &SlidingWindow::Top() const {
        return top;
    }

    const SuffixMapData::iterator &SlidingWindow::Bottom() const {
        return bottom;
    }

    int SlidingWindow::QueryNumberOfDifferentSources() const {
        int count = 0;
        for(const auto& entry : sourceCount)
            count += entry.second > 0 ? 1 : 0;
        return count;
    }

    SuffixMapData::iterator SlidingWindow::QueryLCSInRange() const {
        if (currentSize == 0)
            return end;

        auto it = std::next(top, 1);
        auto lcs = it;

        while(it != bottom){
            if (it->second.SharedChars < lcs->second.SharedChars)
                lcs = it;
            ++it;
        }

        return lcs == bottom ? end : lcs;
    }

    bool SlidingWindow::FitTopToContain(int desiredSources) {
        int sourcesInRange = QueryNumberOfDifferentSources();
        if (sourcesInRange < desiredSources)
            return false;

        while(sourcesInRange >= desiredSources) {
            DecreaseFromTop();
            sourcesInRange = QueryNumberOfDifferentSources();
        }
        return IncreaseFromTop();
    }
}
