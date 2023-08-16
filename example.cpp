
#include "Neural_Network.h"
#include <iostream>

int main(){
    std::vector<Layer> my_layers = {Layer(2, Activation_Function::RELU), Layer(2, Activation_Function::RELU), Layer(2, Activation_Function::RELU)};
    Neural_Network model("Test Model", my_layers);
    for(int layer = 1; layer < 3; layer++){
        for(int i = 0; i < 2; i++){
            for(int j = 0; j < 2; j++){
                model.set_weight(layer, i, j, 1);
            }
        }
    }
    std::unique_ptr<Dataset> test_data1 = std::make_unique<Dataset>(Dataset({{5, 5}, {6, 6}}, {} ,{{5, 5}, {6, 6}}));
    model.test_network(std::move(test_data1));
    std::unique_ptr<Dataset> training_data = std::make_unique<Dataset>(Dataset({{1, 1}, {2, 2}, {4, 4}}, {} ,{{1, 1}, {2, 2}, {4, 4}}));
    model.train_network(std::move(training_data), 0.001, 100, 0.0, Regularization(Regularization_Type::NONE), 1, true, Optimizer(Optimizer_Type::NONE));
    std::unique_ptr<Dataset> test_data2 = std::make_unique<Dataset>(Dataset({{5, 5}, {6, 6}}, {} ,{{5, 5}, {6, 6}}));
    model.test_network(std::move(test_data2));
}
