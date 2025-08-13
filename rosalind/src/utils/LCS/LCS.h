#pragma once
#include <vector>
#include <string_view>
#include <map>

std::string FindLargestCommonSubstring(const std::string &input);
std::string FindLargestCommonSubstring(const std::vector<std::string> &input);
std::string FindLargestCommonSubstringAmongSome(const std::vector<std::string> &input, int sharedBy);
