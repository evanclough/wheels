#include "Layer.h"
#include <cmath>
#include <stdexcept>
#include <iostream>

//basic constructor initializes layer with blank nodes of given size
Layer::Layer(int size, Activation_Function activation){
    this->activation = activation;

    //if size is less than 1, throw error
    if(size < 1){
        throw std::invalid_argument("Layer must have size of at least 1.");
    }

    this->size = size;

    //create array of empty nodes
    std::vector<Node> nodes;
    
    for(int i = 0; i < size; i++){
        Node temp = Node();
        temp.bias = 0;
        temp.weights = {};
        nodes.push_back(temp);
    }

    this->nodes = std::make_unique<std::vector<Node>>(nodes);
}

Layer::Layer(const Layer& cpy){
    this->activation = cpy.activation;
    this->size = cpy.size;
    this->nodes = std::make_unique<std::vector<Node>>(*cpy.nodes);
}

int Layer::get_size(){
    return this->size;
}

Activation_Function Layer::get_activation(){
    return this->activation;
}

std::vector<Node> Layer::get_nodes(){
    return *(this->nodes);
}

//evaluates layer, given input and activation function
std::vector<float> Layer::evaluate(std::vector<float> input){
    //check if input and layer have matching dimensions, throw error if not
    if(input.size() != this->nodes->at(0).weights.size()){
        throw std::invalid_argument("Dimensions of input array and weight array must match");
    }

    //output array
    std::vector<float> output;

    for(int i = 0; i < this->size; i++){
        //start with just bias
        float accum = this->nodes->at(i).bias;
        for(int j = 0; j < this->nodes->at(0).weights.size(); j++){
            accum += this->nodes->at(i).weights[j] * input[j];
        }
        output.push_back(accum);
    }
    //apply activation function
    for(int i = 0; i < output.size(); i++){
        switch(this->activation){
            case RELU:
                output[i] = output[i] < 0 ? 0 : output[i];
            break;
            case TANH:
                output[i] = std::tanh(output[i]);
            break;
            case SIGMOID:
                output[i] = 1 / (1 + std::pow(2.71828, -output[i]));
            break;
        }
    }
    return output;
}

//evaluates layer with no activation
std::vector<float> Layer::evaluate_without_activation(std::vector<float> input){
     //check if input and layer have matching dimensions, throw error if not
    if(input.size() != this->nodes->at(0).weights.size()){
        throw std::invalid_argument("Dimensions of input array and weight array must match");
    }

    //output array
    std::vector<float> output;

    for(int i = 0; i < this->size; i++){
        //start with just bias
        float accum = this->nodes->at(i).bias;
        for(int j = 0; j < this->nodes->at(0).weights.size(); j++){
            accum += this->nodes->at(i).weights[j] * input[j];
        }
        output.push_back(accum);
    }
    return output;
}

//sets given weight to given value
void Layer::set_weight(int i, int j, float weight){
    this->nodes->at(i).weights[j] = weight;
}

//sets weight arrays of all nodes to a given size and constant val
void Layer::set_weights(int size, float weight){
    for(int i = 0; i < this->size; i++){
        this->nodes->at(i).weights = std::vector<float>(size, weight);
    }
}

//sets given bias to given value
void Layer::set_bias(int i,float bias){
    this->nodes->at(i).bias = bias;
}

//sets bieases ofa ll nodes to a given val
void Layer::set_biases(float bias){
    for(int i = 0; i < this->size; i++){
        this->nodes->at(i).bias = bias;
    }
}

//sets nodees array to all have a given number of weights initialized to 0, and sets all biases to 0
void Layer::set_default(int size){
    this->set_weights(size, 0);
    this->set_biases(0);
}