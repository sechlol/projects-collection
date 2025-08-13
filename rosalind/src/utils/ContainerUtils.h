#pragma once
#include <queue>
#include <vector>
#include <sstream>


template<typename T>
T pop_front(std::queue<T>& q){
    T element = q.front();
    q.pop();
    return element;
}

template<typename T>
std::string stringify(const std::vector<T>& v){
    std::stringstream out;
    for (size_t i = 0; i<v.size();i++) {
        out << v[i] << " ";
    }
    std::string s = out.str();
    s.pop_back();
    return s;
}

template<typename T>
bool are_equal(const std::vector<T>& v1, const std::vector<T>& v2){
    if (v1.size() != v2.size())
        return false;

    for (size_t i = 0; i<v1.size(); i++)
        if (v1[i] != v2[i])
            return false;
    return true;
}