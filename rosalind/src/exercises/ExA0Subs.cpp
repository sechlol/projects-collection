#include <sstream>
#include "ExA0Subs.h"
#include "../io/IOUtils.h"

bool ExA0Subs::Test() const {
    auto expected = std::vector<size_t>{2, 4, 10};
    auto result = Execute({"GATATATGCATATACTT", "ATAT"});

    if (result.size() != expected.size())
        return false;

    for (auto i = 0; i<expected.size(); i++)
        if (result[i] != expected[i])
            return false;

    return true;
}

StringPair ExA0Subs::GetInput() const {
    auto lines = LoadFileAsStringLines("datasets/subs.txt");
    return {lines[0], lines[1]};
}

std::vector<size_t> ExA0Subs::Execute(const StringPair &input) const {
    return FindAllSubstrings(input.Primary, input.Sub);
}

std::string ExA0Subs::ToString(const std::vector<size_t> &result) const {
    std::stringstream out;
    for(const auto& item : result)
        out << item << " ";
    return out.str();
}
