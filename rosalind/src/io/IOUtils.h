#pragma once
#include <string>
#include <vector>
#include "FastaString.h"

std::string LoadFileAsString(const std::string& filePath);
std::vector<std::string> LoadFileAsStringLines(const std::string& filePath);
std::vector<FastaString> ParseFastaFile(const std::string& filePath);
void WriteToFile(const std::string& filePath, const std::string& content);
