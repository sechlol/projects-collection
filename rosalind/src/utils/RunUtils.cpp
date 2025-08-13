#pragma once
#include "RunUtils.h"

std::chrono::time_point<std::chrono::high_resolution_clock> TimeNow(){
	return std::chrono::high_resolution_clock::now();
}

double ToMilliseconds(std::chrono::duration<double, std::milli> duration){
	return duration.count();
}