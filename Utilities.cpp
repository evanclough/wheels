#include "Utilities.h"

//trims whitespace off of ends of given string
std::string Utilities::trim(std::string input_string){
    int first_non_whitespace = 0, last_non_whitespace;
    while(std::isspace(input_string[first_non_whitespace]) != 0) first_non_whitespace++;
    for(int i = first_non_whitespace; i < input_string.size(); i++){
        last_non_whitespace = std::isspace(input_string[i]) != 0 ? last_non_whitespace : i;
    }
    return input_string.substr(first_non_whitespace, last_non_whitespace + (last_non_whitespace == input_string.size() - 1));
}

//tokenizes string on given delimiter
std::vector<std::string> Utilities::tokenize(std::string input_string){
    std::vector<std::string> tokens;
    std::size_t index;
    while((index = input_string.find(",")) != std::string::npos){
        tokens.push_back(input_string.substr(0, index));
        input_string.erase(0, index + 1);
    } 
    tokens.push_back(input_string);
    return tokens;
}