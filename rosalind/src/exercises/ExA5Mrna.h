// http://rosalind.info/problems/mrna/

#pragma once
#include "ExerciseImpl.h"

class ExA5Mrna : public ExerciseImpl<std::string, unsigned int>{
public:
    [[nodiscard]] bool Test() const override;
protected:
    [[nodiscard]] std::string GetInput() const override;
    [[nodiscard]] unsigned int Execute(const std::string &input) const override;
    [[nodiscard]] std::string ToString(const unsigned int &result) const override;
};


