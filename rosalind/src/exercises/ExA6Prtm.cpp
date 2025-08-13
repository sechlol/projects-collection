#include "ExA6Prtm.h"
#include "../io/IOUtils.h"
#include "../utils/MassTable.h"

bool ExA6Prtm::Test() const {
    const double expected = 821.392;
    auto result = Execute("SKADYEK");
    return std::abs(result - expected) < 0.0001;
}

std::string ExA6Prtm::GetInput() const {
    return LoadFileAsString("datasets/prtm.txt");
}

double ExA6Prtm::Execute(const std::string &input) const {
    double mass = 0;
    for(const char& c : input){
        mass += MonoisotopicMassTable.at(c);
    }
    return mass;
}

std::string ExA6Prtm::ToString(const double &result) const {
    return std::to_string(result);
}
