#include <sstream>
#include "Ex3Revc.h"
#include "../io/IOUtils.h"

bool Ex3Revc::Test() const {
    return Execute("AAAACCCGGT") == "ACCGGGTTTT";
}

std::string Ex3Revc::GetInput() const {
    return LoadFileAsString("datasets/revc.txt");
}

std::string Ex3Revc::Execute(const std::string &input) const {
    std::stringstream outStream;
    for (auto i = input.size(); i > 0; i--){
        switch (input[i-1])
        {
            case 'A':
                outStream << 'T';
                break;
            case 'T':
                outStream << 'A';
                break;
            case 'C':
                outStream << 'G';
                break;
            case 'G':
                outStream << 'C';
                break;
            default:
                break;
        }
    }
    return outStream.str();
}

std::string Ex3Revc::ToString(const std::string &result) const {
    return result;
}
