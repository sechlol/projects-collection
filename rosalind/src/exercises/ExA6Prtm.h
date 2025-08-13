// http://rosalind.info/problems/prtm/

#pragma once
#include "ExerciseImpl.h"

class ExA6Prtm : public ExerciseImpl<std::string, double>{
public:
    [[nodiscard]] bool Test() const override;
protected:
    [[nodiscard]] std::string GetInput() const override;
    [[nodiscard]] double Execute(const std::string &input) const override;
    [[nodiscard]] std::string ToString(const double &result) const override;
};


