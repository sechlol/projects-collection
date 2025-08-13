#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include "ExA2Grph.h"
#include "../io/IOUtils.h"

const int OVERLAP_LENGTH = 3;

bool ExA2Grph::Test() const {
    auto in = std::vector<FastaString>{
            {"Rosalind_0498", "AAATAAA" },
            {"Rosalind_2391", "AAATTTT" },
            {"Rosalind_2323", "TTTTCCC" },
            {"Rosalind_0442", "AAATCCC" },
            {"Rosalind_5013", "GGGTGGG" },
    };
    auto expected = std::vector<StringViewPair>{
            {"Rosalind_0498", "Rosalind_2391"},
            {"Rosalind_0498", "Rosalind_0442"},
            {"Rosalind_2391", "Rosalind_2323"},
    };

    auto result = Execute(in);
    if (result.size() != expected.size())
        return false;

    for(const auto& r : result){
        bool found = false;
        int i = 0;
        while(!found && i<expected.size())
        {
            if (r.first == expected[i].first && r.second == expected[i].second)
                found = true;
            i++;
        }
        if(!found)
            return false;
    }
    return true;
}

std::vector<FastaString> ExA2Grph::GetInput() const {
    return ParseFastaFile("datasets/grph.txt");
}

std::vector<StringViewPair> ExA2Grph::Execute(const std::vector<FastaString> &input) const {
    std::unordered_map<std::string, std::unordered_set<std::string_view>> suffixGroups;
    std::unordered_map<std::string, std::unordered_set<std::string_view>> prefixGroups;
    std::vector<StringViewPair> adjacency;

    for (const auto& item : input){
        auto size = item.Sequence().size();
        auto prefix = std::string{item.Sequence().substr(0, OVERLAP_LENGTH)};
        auto suffix = std::string{item.Sequence().substr(size-OVERLAP_LENGTH, OVERLAP_LENGTH)};

        auto matchingPrefixGroup = prefixGroups.find(suffix);
        auto matchingSuffixGroup = suffixGroups.find(prefix);

        // current suffix matches some other prefix (item.sequence --> [..items..])
        if (matchingPrefixGroup != prefixGroups.end())
        {
            for (const auto& adjacentStringId : matchingPrefixGroup->second){
                adjacency.emplace_back(item.Id(), adjacentStringId);
            }
        }

        // some other suffix match with current prefix ([..items..] --> item.sequence)
        if (matchingSuffixGroup != suffixGroups.end()){
            for (const auto& adjacentStringId : matchingSuffixGroup->second)
                adjacency.emplace_back(adjacentStringId, item.Id());
        }

        prefixGroups[prefix].insert(item.Id());
        suffixGroups[suffix].insert(item.Id());
    }

    return adjacency;
}

std::string ExA2Grph::ToString(const std::vector<StringViewPair> &result) const {
    std::stringstream out;

    for (const auto& pair : result)
        out << pair.first << " " << pair.second << '\n';

    return out.str();
}

