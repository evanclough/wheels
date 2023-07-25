#include "Neural_Network.h"
#include <stdexcept>

//basic constructor takes a model name and a layer array to create network
Neural_Network::Neural_Network(std::string model_name, std::unique_ptr<std::vector<Layer>> layers){
    //check if layers array has at least two layers, throw error otherwise.
    if(layers->size() < 2){
        throw std::invalid_argument("Layers array must include at least two layers.");
    }

    //initialize weights and biases in layers to 0
    for(int i = 1; i < this->num_layers; i++){
        for(int j = 0; j < this->layers->at(i).get_size(); j++){
            this->layers->at(i).set_default(this->layers->at(i - 1).get_size());
        }
    }

    //set model name, if left blank, just set to model
    this->model_name = model_name == "" ? "Model" : model_name;
    this->num_features = layers->at(0).get_size();
    this->layers = std::move(layers);
}

//runs neural network with given input data and an activation function
std::vector<float> Neural_Network::inference(std::vector<float> input, Activation_Function activation){
    std::vector<float> temp = input;
    for(int i = 1; i < this->num_layers; i++){
        temp = this->layers->at(i).evaluate(temp, activation);
    }
    return temp;
}