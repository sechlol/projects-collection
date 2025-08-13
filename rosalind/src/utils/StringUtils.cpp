#include "StringUtils.h"

std::vector<size_t> FindAllSubstrings(std::string_view primary, std::string_view sub){
    std::vector<size_t> positions;
    size_t foundIndex = 0;

    while(true){
        foundIndex = primary.find(sub, foundIndex);
        if (foundIndex == std::string_view::npos)
            break;

        positions.push_back(foundIndex + 1);
        foundIndex ++;
    }

    return positions;
}

std::vector<size_t> FindAllSubstrings(std::string_view input, int subLength, bool comparer(std::string_view)){
    std::vector<size_t> out;
    for (size_t i = 0; i < input.size()-subLength; i++){
        if(comparer(input.substr(i, subLength)))
            out.push_back(i + 1);
    }
    return out;
}