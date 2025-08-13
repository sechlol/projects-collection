#include <sstream>
#include "ExA3Mprt.h"
#include "../io/IOUtils.h"
#include "../utils/ContainerUtils.h"

const int SUB_LENGTH = 4;

bool ExA3Mprt::Test() const {
    // TODO: Create proper test case
    return true;
}

/* A protein motif is represented by a shorthand as follows:
 * - [XY] means "either X or Y"
 * - {X} means "any amino acid except X."
 * For example, the N-glycosylation motif is written as N{P}[ST]{P}. */
bool Comparer(std::string_view in){
    return
        in[0] == 'N' &&
        in[1] != 'P' &&
        (in[2] == 'S' || in[2] == 'T') &&
        in[3] != 'P';
}

std::vector<FastaString> ExA3Mprt::GetInput() const {
    return ParseFastaFile("datasets/mprt.txt");
}

std::vector<ExA3Out> ExA3Mprt::Execute(const std::vector<FastaString> &input) const {
    std::vector<ExA3Out> out;
    for (const auto& item : input){
        auto pos = FindAllSubstrings(item.Sequence(), SUB_LENGTH, Comparer);
        auto id = item.Id().data();
        if (!pos.empty())
            out.emplace_back(id, pos);
    }
    return out;
}

std::string ExA3Mprt::ToString(const std::vector<ExA3Out> &result) const {
    std::stringstream out;

    for (const auto& item : result) {
        out << item.Id << '\n';
        out << stringify(item.Occurrences) << '\n';
    }

    return out.str();
}