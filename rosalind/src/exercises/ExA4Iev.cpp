#include "ExA4Iev.h"

const int NUMBER_OF_OFFSPRINGS = 2;
const std::vector<RandomVariable> TEST_INPUT {
        {1, 1},
        {0, 1},
        {0, 1},
        {1, 0.75},
        {0, 0.5},
        {1, 0},
};

bool ExA4Iev::Test() const {
    const double expected = 3.5;
    double result = Execute(TEST_INPUT);
    return std::abs(result - expected) < 0.0001;
}

std::vector<RandomVariable> ExA4Iev::GetInput() const {
    return {
            {18410, 1},
            {19998, 1},
            {16197, 1},
            {18217, 0.75},
            {18499, 0.5},
            {16817, 0},
    };
}

double ExA4Iev::Execute(const std::vector<RandomVariable> &input) const {
    double pDominant = 0;
    for(const auto& entry : input){
        pDominant += entry.first * entry.second * NUMBER_OF_OFFSPRINGS;
    }
    return pDominant;
}

std::string ExA4Iev::ToString(const double &result) const {
    return std::to_string(result);
}
