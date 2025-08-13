#pragma once
#include <string>
#include <string_view>

class FastaString {
private:
    const std::string idStr;
    const std::string sequenceStr;
public:
    FastaString(std::string id, std::string sequence);

    [[nodiscard]] std::string_view Id() const;
    [[nodiscard]] std::string_view Sequence() const;
    [[nodiscard]] std::string Raw() const;
};