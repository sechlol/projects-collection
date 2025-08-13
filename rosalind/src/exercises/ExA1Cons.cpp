#include <sstream>
#include "ExA1Cons.h"
#include "../io/IOUtils.h"
#include "../utils/ContainerUtils.h"

bool ExA1Cons::Test() const {
    auto in = std::vector<FastaString>{
        { "Rosalind_1", "ATCCAGCT" },
        { "Rosalind_2", "GGGCAACT" },
        { "Rosalind_3", "ATGGATCT" },
        { "Rosalind_4", "AAGCAACC" },
        { "Rosalind_5", "TTGGAACT" },
        { "Rosalind_6", "ATGCCATT" },
        { "Rosalind_7", "ATGGCACT" },
    };
    ConsensusAndProfile expected(
            "ATGCAACT",
            {
                    5, 1, 0, 0, 5, 5, 0, 0,
                    0, 0, 1, 4, 2, 0, 6, 1,
                    1, 1, 6, 3, 0, 1, 0, 0,
                    1, 5, 0, 0, 0, 1, 1, 6
            },
            4,
            8
    );
    std::string expectedStr = "ATGCAACT\nA: 5 1 0 0 5 5 0 0\nC: 0 0 1 4 2 0 6 1\nG: 1 1 6 3 0 1 0 0\nT: 1 5 0 0 0 1 1 6\n";
    auto result = Execute(in);

    return
        result.Consensus == expected.Consensus &&
        are_equal(result.A(), expected.A()) &&
        are_equal(result.G(), expected.G()) &&
        are_equal(result.C(), expected.C()) &&
        are_equal(result.T(), expected.T()) &&
        ToString(expected) == expectedStr &&
        ToString(result) == expectedStr;
}

std::vector<FastaString> ExA1Cons::GetInput() const {
    return ParseFastaFile("datasets/cons.txt");
}

ConsensusAndProfile ExA1Cons::Execute(const std::vector<FastaString> &input) const {
    std::string consensus;
    size_t rows = input.size();
    size_t cols = input[0].Sequence().size();
    std::vector<int> profile(cols*4);

    for (size_t i = 0; i<cols;i++){
        char maxC = 'A';
        size_t maxPosition = i;
        for (size_t j = 0; j<rows; j++){
            char c = input[j].Sequence()[i];
            size_t position;
            if (c == 'A') position = i;
            else if (c == 'C') position = cols + i;
            else if (c == 'G') position = cols*2 + i;
            else /*if (c == 'T')*/ position = cols*3 + i;

            profile[position]++;
            if (profile[position] > profile[maxPosition]){
                maxC = c;
                maxPosition = position;
            }
        }
        consensus += maxC;
    }

    return {consensus, profile, rows, cols};
}

std::string ExA1Cons::ToString(const ConsensusAndProfile &result) const {
    std::stringstream out;
    out << result.Consensus << '\n';
    out << "A: " << stringify(result.A()) << '\n';
    out << "C: " << stringify(result.C()) << '\n';
    out << "G: " << stringify(result.G()) << '\n';
    out << "T: " << stringify(result.T()) << '\n';
    return out.str();
}
