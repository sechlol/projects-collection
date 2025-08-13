// http://rosalind.info/problems/rna/

#pragma once
#include "ExerciseImpl.h"

class Ex2Rna : public ExerciseImpl<std::string_view, std::string> {
public:
    [[nodiscard]] bool Test() const override;

protected:
    [[nodiscard]] std::string_view GetInput() const override;
    [[nodiscard]] std::string Execute(const std::string_view &input) const override;
    [[nodiscard]] std::string ToString(const std::string &result) const override;
};


