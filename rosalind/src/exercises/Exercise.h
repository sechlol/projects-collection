#pragma once
#include <string_view>
#include "Result.h"

class Exercise {
public:
    [[nodiscard]] virtual Result Run() const = 0;
    [[nodiscard]] virtual bool Test() const;
    [[nodiscard]] virtual Result RunAndCacheInput() = 0;
    virtual ~Exercise();
};

