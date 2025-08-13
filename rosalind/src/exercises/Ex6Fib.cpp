#include "Ex6Fib.h"

const int ITERATIONS = 33;
const int GROWTH_FACTOR = 2;
const unsigned int EXACT_ANSWER = 2863311531;

bool Ex6Fib::Test() const {
    return Execute(GetInput()) == EXACT_ANSWER;
}

GrowthSequence Ex6Fib::GetInput() const {
    return {ITERATIONS, GROWTH_FACTOR};
}

unsigned int Ex6Fib::Execute(const GrowthSequence &input) const {
    unsigned int adultPairs = 0;
    unsigned int newbornPairs = 1;

    for(int i=0; i<input.Iterations-1; i++)
    {
        auto newPairs = adultPairs * input.GrowthFactor;
        adultPairs += newbornPairs;
        newbornPairs = newPairs;
    }

    return adultPairs + newbornPairs;
}

std::string Ex6Fib::ToString(const unsigned int &result) const {
    return std::to_string(result);
}
