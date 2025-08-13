#pragma once
#include <vector>
#include <algorithm>

template <typename T>
class VectorView {
private:
    const T* begin;
    size_t size = 0;
    VectorView(const T* iBegin, size_t iSize);
public:
    VectorView() = default;
    VectorView(const std::vector<T>& in);
    VectorView(const VectorView<T>& other);

    const T* Begin() const;
    const T* End() const;
    size_t Size() const;
    VectorView<T> subVector(size_t count) const;
    VectorView<T> subVector(size_t start, size_t count) const;

    operator std::vector<T>() const;
    const T& operator[](size_t p) const;
    VectorView<T>& operator=(const VectorView<T>& other);
    bool operator<(const VectorView &other) const;
    bool operator>(const VectorView &other) const;
    bool operator<=(const VectorView &other) const;
    bool operator>=(const VectorView &other) const;
    bool operator==(const VectorView &other) const;
    bool operator!=(const VectorView &other) const;
};

template<typename T>
VectorView<T>::VectorView(const std::vector<T> &in) :
        begin(&*in.begin()),
        size(in.size())
{}

template<typename T>
VectorView<T>::VectorView(const T* iBegin, size_t iSize) :
        begin(iBegin),
        size(iSize)
{}

template<typename T>
VectorView<T>::VectorView(const VectorView<T> &other) :
        begin(other.begin),
        size(other.size)
{}

template<typename T>
bool VectorView<T>::operator==(const VectorView &other) const {
    if (this == &other || (begin == other.begin && size == other.size))
        return true;

    if (size != other.size)
        return false;

    for (size_t i = 0; i < size; i++){
       if (*(begin+i) != *(other.begin+i))
           return false;
    }
    return true;
}

template<typename T>
bool VectorView<T>::operator!=(const VectorView &other) const {
    return !(other == *this);
}

template<typename T>
bool VectorView<T>::operator<(const VectorView &other) const {
    size_t minSize = std::min(size, other.size);
    size_t count = 0;
    auto it = begin;
    auto otherIt = other.begin;

    for(size_t i = 0; i < minSize; i++){
        if (*it < *otherIt)
            return true;
        else if (*it > *otherIt)
            return false;
        it++;
        otherIt++;
    }
    // At this point they're equal. If this->size < other.size,
    // it means it is lexicographically less than other.
    return size < other.size;
}

template<typename T>
bool VectorView<T>::operator>(const VectorView &other) const {
    return other < *this;
}

template<typename T>
bool VectorView<T>::operator<=(const VectorView &other) const {
    return !(other < *this);
}

template<typename T>
bool VectorView<T>::operator>=(const VectorView &other) const {
    return !(*this < other);
}

template<typename T>
VectorView<T> VectorView<T>::subVector(size_t count) const {
    return subVector(0, count);
}

template<typename T>
VectorView<T> VectorView<T>::subVector(size_t start, size_t count) const {
    if (start >= size)
        return VectorView<T>(begin, 0);

    if (start + count > size)
        count = size - start;

    auto s = begin + start;
    return VectorView<T>(s, count);
}

template<typename T>
const T& VectorView<T>::operator[](size_t p) const {
    return (*(begin + p));
}

template<typename T>
VectorView<T>& VectorView<T>::operator=(const VectorView<T> &other) {
    if (&other != this) {
        begin = other.begin;
        size = other.size;
    }
    return *this;
}

template<typename T>
VectorView<T>::operator std::vector<T>() const {
    return {begin, begin + size};
}

template<typename T>
size_t VectorView<T>::Size() const {
    return size;
}

template<typename T>
const T* VectorView<T>::Begin() const {
    return begin;
}

template<typename T>
const T* VectorView<T>::End() const {
    return begin + size;
}
