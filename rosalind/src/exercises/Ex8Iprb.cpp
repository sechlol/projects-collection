#include "Ex8Iprb.h"

const int HOMOZYGOUS_DOMINANT = 27;
const int HETEROZYGOUS = 19;
const int HOMOZYGOUS_RECESSIVE = 30;
const double EXACT_ANSWER = 0.732368f;

bool Ex8Iprb::Test() const {
    auto answer = Execute(GetInput());
    return std::abs(answer - EXACT_ANSWER) < 0.000001;
}

Alleles Ex8Iprb::GetInput() const {
    return {HOMOZYGOUS_DOMINANT, HETEROZYGOUS, HOMOZYGOUS_RECESSIVE};
}

double Ex8Iprb::Execute(const Alleles &input) const {
    double hd(input.HomoDominant);
    double he(input.Hetero);
    double hr(input.HomoRecessive);

    auto total = hd + he + hr;

    // 100% recessive phenotype (recessive + recessive)
    double pFullRecessive = hr/total * ((hr-1)/(total-1));

    // 25% recessive phenotype (hetero + hetero)
    double pTwoHetero = (he/total) * ((he-1)/(total-1));

    // 50% recessive phenotype
    double pHeteroRecessive =
            (he/total) * (hr/(total-1)) + // hetero + recessive
            (hr/total) * (he/(total-1)); // recessive + hetero

    double pRecessive = pFullRecessive + pHeteroRecessive * 0.5f + pTwoHetero * 0.25f;
    return 1-pRecessive;
}

std::string Ex8Iprb::ToString(const double &result) const {
    return std::to_string(result);
}
