# pragma once
#include <fstream>
#include <sstream>
#include "IOUtils.h"

const std::string PROJECT_ROOT = "../../";

std::string LoadFileAsString(const std::string& filePath){
	std::ifstream file(PROJECT_ROOT+filePath);
	if (!file)
		return "";

	std::ostringstream outStream;

	outStream << file.rdbuf();
	return outStream.str();
}

std::vector<std::string> LoadFileAsStringLines(const std::string& filePath){
	std::ifstream file(PROJECT_ROOT+filePath);
	std::string line;
	std::vector<std::string> linesList;

	if (!file)
		return linesList;

	while(file >> line)
		linesList.push_back(line);

	return linesList;
}

void WriteToFile(const std::string& filePath, const std::string& content){
	std::ofstream file(PROJECT_ROOT+filePath, std::ofstream::out);
	file << content;
}

std::vector<FastaString> ParseFastaFile(const std::string& filePath){
    std::ifstream file(PROJECT_ROOT+filePath);

    if (!file){
        return {};
    }

    std::string id, sequence, line;
    std::vector<FastaString> fastaSequences;
    while(file >> line){
        if(line[0] == '>'){
            if (!id.empty()){
                fastaSequences.emplace_back(id, sequence);
                sequence = "";
            }
            id = line.erase(0, 1);
        }
        else{
            sequence += line;
        }
    }

    fastaSequences.emplace_back(id, sequence);
    return fastaSequences;
}