//hthe neural network class allows for creation and training of a simple neural network

#include <vector>
#include <string>
#include <memory>
#include "Layer.h"


class Neural_Network {
    public:
        int num_layers, num_features;
        std::string model_name;
        std::unique_ptr<std::vector<Layer>> layers;
    private:
        //default constructor
        Neural_Network(std::string model_name, std::unique_ptr<std::vector<Layer>> layers);

        //runs neural network with a set of input data and returns output
        std::vector<float> inference(std::vector<float> input);
}; 