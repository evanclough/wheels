//hthe neural network class allows for creation and training of a simple neural network

#include <vector>
#include <string>
#include <memory>
#include "Layer.h"
#include "Dataset.h"

class Neural_Network {
    private:
        int num_layers, num_features;
        std::string model_name;
        std::unique_ptr<std::vector<Layer>> layers;
    public:
        //default constructor
        Neural_Network(std::string model_name, std::unique_ptr<std::vector<Layer>> layers);

        //runs neural network with a set of input data and returns output
        std::vector<float> inference(std::vector<float> input, Activation_Function activation);

        //runs backpropogation on network given feature and label vectors
        void backprop(std::vector<float> feature, std::vector<float> labels, Activation_Function activation);

        //trains network with a given training dataset, learning rate, number of epochs, and validation split
        void train_network(Dataset training_data, float learning_rate, int epochs, float validation_split);

        //tests network on given test data set and returns error
        void test_network(Dataset test_data);
}; 