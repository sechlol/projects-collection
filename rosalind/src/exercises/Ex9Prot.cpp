#include <sstream>
#include "Ex9Prot.h"
#include "../io/IOUtils.h"
#include "../utils/CodonTable.h"

const std::string_view TEST_INPUT {"AUGGCCAUGGCGCCCAGAACUGAGAUCAAUAGUACCCGUAUUAACGGGUGA"};
const std::string_view EXACT_ANSWER {"MAMAPRTEINSTRING"};

bool Ex9Prot::Test() const {
    return Execute(TEST_INPUT.data()) == EXACT_ANSWER;
}

std::string Ex9Prot::GetInput() const {
    return LoadFileAsString("datasets/prot.txt");
}

std::string Ex9Prot::Execute(const std::string &input) const {
    size_t index = 0;
    std::string proteinString;

    while (true)
    {
        auto protein = input.substr(index, 3);
        auto codon = CodonTable.at(protein);
        if (codon == STOP_MARKER){
            break;
        }
        else{
            proteinString += codon;
            index += 3;
        }
    }

    return proteinString;
}

std::string Ex9Prot::ToString(const std::string &result) const {
    return result;
}
