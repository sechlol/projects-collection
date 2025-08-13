// http://rosalind.info/problems/lcsm/

#pragma once
#include <vector>
#include "ExerciseImpl.h"
#include "../io/FastaString.h"

class ExA7Lcsm : public ExerciseImpl<std::vector<std::string>, std::string>{
public:
    [[nodiscard]] bool Test() const override;
protected:
    [[nodiscard]] std::vector<std::string> GetInput() const override;
    [[nodiscard]] std::string Execute(const std::vector<std::string> &input) const override;
    [[nodiscard]] std::string ToString(const std::string &result) const override;
};


