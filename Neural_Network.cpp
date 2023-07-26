#include "Neural_Network.h"
#include <stdexcept>

//basic constructor takes a model name and a layer array to create network
Neural_Network::Neural_Network(std::string model_name, std::unique_ptr<std::vector<Layer>> layers){
    //check if layers array has at least two layers, throw error otherwise.
    if(layers->size() < 2){
        throw std::invalid_argument("Layers array must include at least two layers.");
    }
    this->layers = std::move(layers);
    this->num_layers = this->layers->size();
    this->num_features = this->layers->at(0).get_size();

    //initialize weights and biases in layers to 0
    for(int i = 1; i < this->num_layers; i++){
        this->layers->at(i).set_default(this->layers->at(i - 1).get_size());
    }
    this->model_name = model_name == "" ? "Model" : model_name;
}

//runs neural network with given input data and an activation function
std::vector<float> Neural_Network::inference(std::vector<float> input, Activation_Function activation){
    std::vector<float> temp = input;
    for(int i = 1; i < this->num_layers; i++){
        temp = this->layers->at(i).evaluate(temp, activation);
    }
    return temp;
}

//runs backpropogation on network given feature and label vectors
void Neural_Network::backprop(std::vector<float> feature, std::vector<float> labels, Activation_Function activation){
    ;
}

//trains network with a given training dataset, learning rate, number of epochs, and validation split
void Neural_Network::train_network(Dataset training_data, float learning_rate, int epochs, float validation_split){
    ;
}

//tests network on given test data set and returns error
void Neural_Network::test_network(Dataset test_data){
    ;
}