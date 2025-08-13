#include "FastaString.h"
#include <utility>

FastaString::FastaString(std::string  id, std::string  sequence) :
    idStr(std::move(id)),
    sequenceStr(std::move(sequence))
{}

std::string_view FastaString::Id() const {
    return idStr;
}

std::string_view FastaString::Sequence() const {
    return sequenceStr;
}

std::string FastaString::Raw() const {
    return sequenceStr;
}
