#include "Ex5Gc.h"
#include "../io/IOUtils.h"

bool Ex5Gc::Test() const {
    auto in = std::vector<FastaString>{
        { "Rosalind_6404", "CCTGCGGAAGATCGGCACTAGAATAGCCAGAACCGTTTCTCTGAGGCTTCCGGCCTTCCCTCCCACTAATAATTCTGAGG" },
        { "Rosalind_5959", "CCATCGGTAGCGCATCCTTAGTCCAATTAAGTCCCTATCCAGGCGCTCCGCCGAAGGTCTATATCCATTTGTCAGCAGACACGC" },
        { "Rosalind_0808", "CCACCCTCGTGGTATGGCTAGGCATTCAGGAACCGGAGAACGCTTCAGACCAGCCCGGACTGGGAACCTGCGGGCAGTAGGTGGAAT" },
    };
    auto out = Execute(in);

    return
        out.first == "Rosalind_0808" &&
        out.second > 60.919539 && out.second < 60.919541;
}

std::vector<FastaString> Ex5Gc::GetInput() const {
    auto barnabo = ParseFastaFile("datasets/fasta.txt");
    return barnabo;
}

std::pair<std::string_view, double> Ex5Gc::Execute(const std::vector<FastaString> &input) const {
    double maxGCContent = 0;
    int indexMax = 0;
    for (size_t i = 0; i<input.size(); i++){
        double newGC = CalculateGCContent(input[i].Sequence());
        if (newGC > maxGCContent){
            maxGCContent = newGC;
            indexMax = i;
        }
    }
    return {input[indexMax].Id(), maxGCContent};
}

double Ex5Gc::CalculateGCContent(const std::string_view& sequence) {
    int size = sequence.size();
    int gc = 0;
    for(const auto& symbol : sequence){
        if (symbol == 'G' || symbol == 'C')
            gc++;
    }
    return 100*(double)gc/(double)size;
}

std::string Ex5Gc::ToString(const std::pair<std::string_view, double> &result) const {
    return std::string(result.first) + '\n' + std::to_string(result.second);
}