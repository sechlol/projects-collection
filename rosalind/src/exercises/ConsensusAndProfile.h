#pragma once
#include <vector>
#include <string>

class ConsensusAndProfile{
public:
    ConsensusAndProfile(std::string consensus, std::vector<int> iProfile, size_t iRows, size_t iCols);
    const std::string Consensus;
    [[nodiscard]] std::vector<int> A() const;
    [[nodiscard]] std::vector<int> C() const;
    [[nodiscard]] std::vector<int> G() const;
    [[nodiscard]] std::vector<int> T() const;
private:
    std::vector<int> profile;
    size_t rows{};
    size_t cols{};
};