// http://rosalind.info/problems/hamm/

#pragma once
#include <string>
#include "ExerciseImpl.h"

class Ex4Hamm : public ExerciseImpl<std::string, int>{
public:
    [[nodiscard]] bool Test() const override;
protected:
    [[nodiscard]] std::string GetInput() const override;
    [[nodiscard]] int Execute(const std::string &input) const override;
    [[nodiscard]] std::string ToString(const int &result) const override;
};


