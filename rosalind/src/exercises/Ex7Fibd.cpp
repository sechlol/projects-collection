#include <queue>
#include "Ex7Fibd.h"
#include "../utils/ContainerUtils.h"

const int ITERATIONS = 92;
const int GROWTH_FACTOR = 1;
const int LIFESPAN = 19;
const unsigned long long EXACT_ANSWER = 7513165195107624104;

bool Ex7Fibd::Test() const {
    return Execute(GetInput()) == EXACT_ANSWER;
}

EvolutionSequence Ex7Fibd::GetInput() const {
    return {ITERATIONS, GROWTH_FACTOR, LIFESPAN};
}

unsigned long long int Ex7Fibd::Execute(const EvolutionSequence &input) const {
    std::queue<unsigned long long> newbornRecord;
    unsigned long long adults = 0;
    unsigned long long babies = 0;

    for (int i=0; i<input.Iterations; i++){
        // note: adults reproduce before dying
        auto newborns = i == 0 ? 1 : adults * input.GrowthFactor;
        auto dead = newbornRecord.size() == input.Lifespan ? pop_front(newbornRecord) : 0;

        adults = adults - dead + babies;
        babies = newborns;
        newbornRecord.push(newborns);
    }

    return adults + babies;
}

std::string Ex7Fibd::ToString(const unsigned long long int &result) const {
    return std::to_string(result);
}
