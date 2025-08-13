// http://rosalind.info/problems/subs/

#pragma once
#include <vector>
#include "ExerciseImpl.h"
#include "../utils/StringUtils.h"

class ExA0Subs : public ExerciseImpl<StringPair, std::vector<size_t>>{
public:
    [[nodiscard]] bool Test() const override;
protected:
    [[nodiscard]] StringPair GetInput() const override;
    [[nodiscard]] std::vector<size_t> Execute(const StringPair &input) const override;
    [[nodiscard]] std::string ToString(const std::vector<size_t> &result) const override;
};


