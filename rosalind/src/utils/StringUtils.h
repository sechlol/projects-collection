#pragma once
#include <string>
#include <string_view>
#include <vector>

typedef std::pair<std::string_view, std::string_view> StringViewPair;

struct StringPair{
public:
    std::string Primary;
    std::string Sub;
};

std::vector<size_t> FindAllSubstrings(std::string_view input, int subLength, bool comparer(std::string_view));
std::vector<size_t> FindAllSubstrings(std::string_view primary, std::string_view sub);
