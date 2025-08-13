// http://rosalind.info/problems/cons/

#pragma once
#include <vector>
#include "ExerciseImpl.h"
#include "../io/FastaString.h"
#include "ConsensusAndProfile.h"

class ExA1Cons : public ExerciseImpl<std::vector<FastaString>, ConsensusAndProfile>{
public:
    [[nodiscard]] bool Test() const override;
protected:
    [[nodiscard]] std::vector<FastaString> GetInput() const override;
    [[nodiscard]] ConsensusAndProfile Execute(const std::vector<FastaString> &input) const override;
    [[nodiscard]] std::string ToString(const ConsensusAndProfile &result) const override;
};
