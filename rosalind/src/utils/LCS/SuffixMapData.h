#pragma once
#include <string_view>
#include <map>

namespace LCS{

struct Prefix {
    int ParentIndex;
    unsigned int SharedChars;
};

typedef std::map<std::string_view, Prefix> SuffixMapData;
}
