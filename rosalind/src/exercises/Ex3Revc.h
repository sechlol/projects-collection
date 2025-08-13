// http://rosalind.info/problems/revc/

#pragma once
#include <string>
#include "ExerciseImpl.h"

class Ex3Revc : public ExerciseImpl<std::string, std::string>{
public:
    [[nodiscard]] bool Test() const override;
protected:
    [[nodiscard]] std::string GetInput() const override;
    [[nodiscard]] std::string Execute(const std::string &input) const override;
    [[nodiscard]] std::string ToString(const std::string &result) const override;
};


