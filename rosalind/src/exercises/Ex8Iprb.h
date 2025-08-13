// http://rosalind.info/problems/iprb/

#pragma once
#include "ExerciseImpl.h"

struct Alleles{
public:
    int HomoDominant = 0;
    int Hetero = 0;
    int HomoRecessive = 0;
};

class Ex8Iprb : public ExerciseImpl<Alleles, double> {
public:
    [[nodiscard]] bool Test() const override;
protected:
    [[nodiscard]] Alleles GetInput() const override;
    [[nodiscard]] double Execute(const Alleles &input) const override;
    [[nodiscard]] std::string ToString(const double &result) const override;
};