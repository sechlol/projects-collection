#include <algorithm>
#include <utility>
#include "SuffixMap.h"

using namespace std;

namespace LCS {

    SuffixMap::SuffixMap(string&& input, vector<size_t>&& sourceSizes) :
        source(std::move(input)),
        sourceView(source),
        sizesOfSources(std::move(sourceSizes)),
        data(CreateSuffixMapData(sourceView, sizesOfSources))
    {}

    SuffixMapData SuffixMap::CreateSuffixMapData(string_view input, const vector<size_t> &sizes) {
        SuffixMapData newData;
        int parentIndex = 0;
        int count = 0;
        for(const auto& size : sizes){
            for (int i = 0; i < size; i++){
                auto suffix = input.substr(count++);
                newData[suffix] = {parentIndex, 0};
            }
            parentIndex++;
        }

        ComputeSuffixArray(newData);
        return newData;
    }

    int SuffixMap::FindCommonPrefixLength(string_view s1, string_view s2) {
        size_t minSize = min(s1.size(), s2.size());
        int i = 0;
        while (i < minSize && s1[i] == s2[i])
            i++;
        return i;
    }

    void SuffixMap::ComputeSuffixArray(SuffixMapData& mapData) {
        auto it = mapData.begin();

        // Skips the first entry as it doesn't have other entries to be compared with
        auto previousEntry = it++;

        while (it != mapData.end()) {
            it->second.SharedChars = FindCommonPrefixLength(it->first, previousEntry->first);
            previousEntry = it++;
        }
    }
}