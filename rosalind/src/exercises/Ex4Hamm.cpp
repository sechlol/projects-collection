#include "Ex4Hamm.h"
#include "../io/IOUtils.h"

bool Ex4Hamm::Test() const {
    return Execute("GAGCCTACTAACGGGAT\nCATCGTAATGACGGCCT") == 7;
}

std::string Ex4Hamm::GetInput() const {
    return LoadFileAsString("datasets/hamm.txt");
}

int Ex4Hamm::Execute(const std::string &input) const {
    size_t size = input.size();
    size_t firstStringIndex = 0;
    size_t secondStringIndex = input.find('\n') + 1;
    int differences = 0;

    while(secondStringIndex < size){
        if(input[firstStringIndex++] != input[secondStringIndex++])
            differences++;
    }
    return differences;
}

std::string Ex4Hamm::ToString(const int &result) const {
    return std::to_string(result);
}
