
#include "Neural_Network.h"
#include <iostream>

int main(){
    std::unique_ptr<std::vector<Layer>> layers = std::make_unique<std::vector<Layer>>();
    layers->push_back(Layer(5));
    layers->push_back(Layer(4));
    layers->push_back(Layer(3));
    Neural_Network model("Test Model", std::move(layers));
    std::vector<float> res = model.inference({1, 2, 3, 4, 5}, Activation_Function::RELU);
}
