#pragma once
#include <string>
#include <chrono>

std::chrono::time_point<std::chrono::high_resolution_clock> TimeNow();
double ToMilliseconds(std::chrono::duration<double, std::milli> duration);