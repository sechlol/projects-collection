// http://rosalind.info/problems/gc/

#pragma once
#include <string>
#include <vector>
#include "ExerciseImpl.h"
#include "../io/FastaString.h"

class Ex5Gc : public ExerciseImpl<std::vector<FastaString>, std::pair<std::string_view, double>>{
public:
    [[nodiscard]] bool Test() const override;
protected:
    [[nodiscard]] std::vector<FastaString> GetInput() const override;
    [[nodiscard]] std::pair<std::string_view, double> Execute(const std::vector<FastaString> &input) const override;
    [[nodiscard]] std::string ToString(const std::pair<std::string_view, double> &result) const override;
private:
    static double CalculateGCContent(const std::string_view& sequence);
};