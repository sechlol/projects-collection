// http://rosalind.info/problems/mprt/

#pragma once
#include <vector>
#include "ExerciseImpl.h"
#include "../io/FastaString.h"
#include "../utils/StringUtils.h"

struct ExA3Out{
public:
    std::string Id;
    std::vector<size_t> Occurrences{};

    ExA3Out(std::string id, std::vector<size_t> occurrences) :
        Id(std::move(id)),
        Occurrences(std::move(occurrences)) {}
};

class ExA3Mprt : public ExerciseImpl<std::vector<FastaString>, std::vector<ExA3Out>> {
public:
    [[nodiscard]] bool Test() const override;
protected:
    [[nodiscard]] std::vector<FastaString> GetInput() const override;
    [[nodiscard]] std::vector<ExA3Out> Execute(const std::vector<FastaString> &input) const override;
    [[nodiscard]] std::string ToString(const std::vector<ExA3Out> &result) const override;
};


