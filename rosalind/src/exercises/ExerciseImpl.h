#pragma once
#include <optional>
#include "Exercise.h"
#include "Result.h"
#include "../utils/RunUtils.h"

template<typename TIn, typename TOut>
class ExerciseImpl : public Exercise {
public:
    [[nodiscard]] Result Run() const override;
    [[nodiscard]] Result RunAndCacheInput() override;
protected:
    [[nodiscard]] virtual TIn GetInput() const = 0;
    [[nodiscard]] virtual TOut Execute(const TIn& input) const = 0;
    [[nodiscard]] virtual std::string ToString(const TOut& result) const = 0;
private:
    std::optional<TIn> cachedInputValue;
};

/* Must place the implementation here because of the compiler not being able to link template functions
 * with the implementation(s).
 * https://stackoverflow.com/questions/495021/why-can-templates-only-be-implemented-in-the-header-file */
template<typename TIn, typename TOut>
Result ExerciseImpl<TIn, TOut>::Run() const {
    auto input = this->GetInput();
    auto startTime = TimeNow();
    auto result = this->Execute(input);
    auto endTime = TimeNow();
    return {this->ToString(result), ToMilliseconds(endTime-startTime)};
}

template<typename TIn, typename TOut>
Result ExerciseImpl<TIn, TOut>::RunAndCacheInput() {
    if (!cachedInputValue.has_value()){
        cachedInputValue.emplace(this->GetInput());
    }

    auto startTime = TimeNow();
    auto result = this->Execute(cachedInputValue.value());
    auto endTime = TimeNow();
    return {this->ToString(result), ToMilliseconds(endTime-startTime)};
}