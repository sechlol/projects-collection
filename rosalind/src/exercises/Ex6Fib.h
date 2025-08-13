// http://rosalind.info/problems/fib/
#pragma once
#include "ExerciseImpl.h"

struct GrowthSequence{
public:
    int Iterations = 0;
    int GrowthFactor = 0;
};

class Ex6Fib : public ExerciseImpl<GrowthSequence, unsigned int> {
public:
    [[nodiscard]] bool Test() const override;
protected:
    [[nodiscard]] GrowthSequence GetInput() const override;
    [[nodiscard]] unsigned int Execute(const GrowthSequence &input) const override;
    [[nodiscard]] std::string ToString(const unsigned int &result) const override;
};


