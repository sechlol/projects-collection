// http://rosalind.info/problems/grph/

#pragma once

#include <vector>
#include "ExerciseImpl.h"
#include "../io/FastaString.h"
#include "../utils/StringUtils.h"

class ExA2Grph : public ExerciseImpl<std::vector<FastaString>, std::vector<StringViewPair>> {
public:
    [[nodiscard]] bool Test() const override;
protected:
    [[nodiscard]] std::vector<FastaString> GetInput() const override;
    [[nodiscard]] std::vector<StringViewPair> Execute(const std::vector<FastaString> &input) const override;
    [[nodiscard]] std::string ToString(const std::vector<StringViewPair> &result) const override;
};