#include <sstream>
#include <algorithm>
#include "LCS.h"
#include "SuffixMap.h"
#include "SlidingWindow.h"
#include "StringCompactor.h"

using std::vector;
using std::string;
using namespace LCS;

namespace {
    vector<size_t> GetSourcesSizeWithTermination(const vector<string> &sources) {
        vector<size_t> out;
        out.reserve(sources.size());
        for (const auto& str : sources)
            out.push_back(str.size()+1);
        return out;
    }
}
string FindLargestCommonSubstring(const string &input){
    return FindLargestCommonSubstringAmongSome({input}, 1);
}

string FindLargestCommonSubstring(const vector<string> &input){
    return FindLargestCommonSubstringAmongSome(input, input.size());
}

string FindLargestCommonSubstringAmongSome(const vector<string> &input, int sharedBy){
    if (sharedBy > input.size())
        return {};

    StringCompactor compactor({'G', 'T', 'C', 'A'});

    SuffixMap data(
            compactor.EncodeWithTermination(input),
            GetSourcesSizeWithTermination(input));

    // Avoid the bottom part of the map, because it's just full of strings
    // beginning with termination symbols
    SlidingWindow window(data.data.begin(), std::prev(data.data.end(), input.size()));
    auto lcs = data.data.begin();

    while(window.Bottom() != window.End() && window.Top() != window.End()){
        while (window.QueryNumberOfDifferentSources() < sharedBy && window.Bottom() != window.End())
            window.IncreaseFromBottom(); // NOP

       if (window.Size() > sharedBy)
            window.FitTopToContain(sharedBy);

       auto localLcs = window.QueryLCSInRange();
       if(localLcs != window.End() && localLcs->second.SharedChars > lcs->second.SharedChars)
           lcs = localLcs;

       window.DecreaseFromTop();
    }

    return compactor.Decode(string(lcs->first.begin(), std::next(lcs->first.begin(), lcs->second.SharedChars)));
}