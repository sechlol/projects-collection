#pragma once
#include <string>
#include <vector>
#include <unordered_map>

namespace LCS{

class StringCompactor {
private:
    std::unordered_map<char, char> translationTable;
    std::unordered_map<char, char> reverseTable;
    char symbolsCount = 0;
public:
    explicit StringCompactor(std::string_view sample);
    explicit StringCompactor(const std::vector<char>& symbols);

    [[nodiscard]] char GetTerminationSymbol(int forIndex) const;
    [[nodiscard]] std::string Encode(std::string_view in) const;
    [[nodiscard]] std::string Decode(std::string_view in) const;

    [[nodiscard]] std::string EncodeWithTermination(const std::vector<std::string>& in) const;
    //[[nodiscard]] std::vector<std::string> Decode(const std::vector<std::string>& in) const;
};
}