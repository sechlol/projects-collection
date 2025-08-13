#pragma once
#include "SuffixMapData.h"
#include <vector>
#include <map>
#include <string>

namespace LCS{

    class SuffixMap {
    private:
        std::string source;
        std::string_view sourceView;
        std::vector<size_t> sizesOfSources;
    public:
        SuffixMapData data;

        explicit SuffixMap(std::string&& input, std::vector<size_t>&& sourceSizes);
    private:
        static SuffixMapData CreateSuffixMapData(std::string_view input, const std::vector<size_t>& sizes);
        static int FindCommonPrefixLength(std::string_view s1, std::string_view s2) ;
        static void ComputeSuffixArray(SuffixMapData& mapData);
    };
}