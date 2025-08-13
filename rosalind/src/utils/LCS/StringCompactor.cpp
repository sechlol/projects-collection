#include "StringCompactor.h"
using std::vector;
using std::string;
using std::string_view;

namespace LCS{
    StringCompactor::StringCompactor(string_view sample) :
        StringCompactor(vector<char>(sample.begin(), sample.end())){}

    StringCompactor::StringCompactor(const std::vector<char> &symbols) {
        for(const auto& c : symbols) {
            if (translationTable.find(c) != translationTable.end())
                continue;
            translationTable[c] = symbolsCount;
            reverseTable[symbolsCount] = c;
            ++symbolsCount;
        }
    }

    string StringCompactor::Encode(string_view in) const {
        string encoded;
        for (const char& c : in){
            encoded += translationTable.at(c);
        }
        return encoded;
    }

    std::string StringCompactor::Decode(std::string_view in) const {
        string decoded;
        for (const char c : in){
            auto it = reverseTable.find(c);
            if (it != reverseTable.end()) {
                decoded += it->second;
            }
        }
        return decoded;
    }

    string StringCompactor::EncodeWithTermination(const vector<string> &in) const {
        string out;
        for (size_t i = 0; i<in.size(); i++) {
            out += Encode(in[i]);
            out += GetTerminationSymbol(i);
        }
        return out;
    }

    char StringCompactor::GetTerminationSymbol(int forIndex) const {
        return (char)(symbolsCount+forIndex);
    }
}
