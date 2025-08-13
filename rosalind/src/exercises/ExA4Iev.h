// http://rosalind.info/problems/iev/

#pragma once

#include <vector>
#include "ExerciseImpl.h"

typedef std::pair<int, double> RandomVariable;

class ExA4Iev : public ExerciseImpl<std::vector<RandomVariable>, double> {
public:
    [[nodiscard]] bool Test() const override;
protected:
    [[nodiscard]] std::vector<RandomVariable> GetInput() const override;
    [[nodiscard]] double Execute(const std::vector<RandomVariable> &input) const override;
    [[nodiscard]] std::string ToString(const double &result) const override;
};


