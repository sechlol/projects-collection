#include <unordered_map>
#include "ExA7Lcsm.h"
#include "../io/IOUtils.h"
#include "../utils/LCS/LCS.h"

const auto TEST_INPUT = std::vector<std::string>{"GATTACA","CAGATTA","ACATAGATTATATA"};

bool ExA7Lcsm::Test() const {
    return Execute(TEST_INPUT) == "GATTA";
//    return Execute(GetInput()) == "CTTGACTAAGCGTCGATATATCCGTTCTGCATGCGTCAGG";
}

std::vector<std::string> ExA7Lcsm::GetInput() const {
    //return TEST_INPUT; // To speed up tests! Remove this for actual execution
    auto entries = ParseFastaFile("datasets/lcsm.txt");
    std::vector<std::string> out;

    out.reserve(entries.size());
    for(const auto& entry : entries)
        out.emplace_back(std::move(entry.Raw()));
    return out;
}

std::string ExA7Lcsm::Execute(const std::vector<std::string> &input) const {
    return FindLargestCommonSubstring(input);
}

std::string ExA7Lcsm::ToString(const std::string &result) const {
    return result;
}
