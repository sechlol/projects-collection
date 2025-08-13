// http://rosalind.info/problems/fibd/
#pragma once
#include "ExerciseImpl.h"

struct EvolutionSequence{
public:
    int Iterations = 0;
    int GrowthFactor = 0;
    int Lifespan = 0;
};

class Ex7Fibd : public ExerciseImpl<EvolutionSequence, unsigned long long>{
public:
    [[nodiscard]] bool Test() const override;
protected:
    [[nodiscard]] EvolutionSequence GetInput() const override;
    [[nodiscard]] unsigned long long int Execute(const EvolutionSequence &input) const override;
    [[nodiscard]] std::string ToString(const unsigned long long int &result) const override;
};


