#include "ExA5Mrna.h"
#include "../io/IOUtils.h"
#include "../utils/CodonTable.h"

const unsigned int MODULO = 1000000;

bool ExA5Mrna::Test() const {
    std::string input = "MA";
    input.push_back(STOP_MARKER);
    return Execute(input) == 12;
}

std::string ExA5Mrna::GetInput() const {
    std::string input = LoadFileAsString("datasets/mrna.txt");
    input.push_back(STOP_MARKER);
    return input;
}

unsigned int ExA5Mrna::Execute(const std::string &input) const {
    unsigned int sum = 1;
    for (const char& c : input)
        sum = (sum * ReverseCodonTable.at(c).size()) % MODULO;
    return sum;
}

std::string ExA5Mrna::ToString(const unsigned int &result) const {
    return std::to_string(result);
}
