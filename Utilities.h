#include <string>
#include <vector>
#include <iostream>

// the utilities namespace provides some useful functions that can be used across the library
namespace Utilities {
    //trims whitespace off of ends of given string
    std::string trim(std::string input_string);

    //tokenize tokenizes given string about a given delimiter
    std::vector<std::string> tokenize(std::string input_string);
}
