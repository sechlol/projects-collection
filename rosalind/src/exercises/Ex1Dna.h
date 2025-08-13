#pragma once
#include "Exercise.h"
#include "ExerciseImpl.h"

struct SymbolsCount{
public:
    int ACount;
    int CCount;
    int TCount;
    int GCount;
};

class Ex1Dna : public ExerciseImpl<std::string, SymbolsCount> {
public:
    [[nodiscard]] bool Test() const override;

protected:
    [[nodiscard]] std::string GetInput() const override;
    [[nodiscard]] SymbolsCount Execute(const std::string &input) const override;
    [[nodiscard]] std::string ToString(const SymbolsCount &result) const override;
};


