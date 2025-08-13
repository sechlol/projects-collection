#include "ConsensusAndProfile.h"
#include <utility>

ConsensusAndProfile::ConsensusAndProfile(std::string consensus, std::vector<int> iProfile, size_t iRows, size_t iCols) :
    Consensus(std::move(consensus)),
    profile(std::move(iProfile)),
    rows(iRows),
    cols(iCols)
{}

std::vector<int> ConsensusAndProfile::A() const {
    return std::vector<int>(profile.begin(), profile.begin() + cols);
}

std::vector<int> ConsensusAndProfile::C() const {
    return std::vector<int>(profile.begin() + cols, profile.begin() + cols * 2);
}

std::vector<int> ConsensusAndProfile::G() const {
    return std::vector<int>(profile.begin() + cols * 2, profile.begin() + cols * 3);
}

std::vector<int> ConsensusAndProfile::T() const {
    return std::vector<int>(profile.begin() + cols * 3, profile.begin() + cols * 4);
}
