
#include "Neural_Network.h"
#include <iostream>

int main(){
    std::unique_ptr<std::vector<Layer>> layers = std::make_unique<std::vector<Layer>>();
    layers->push_back(Layer(3, Activation_Function::RELU));
    layers->push_back(Layer(4, Activation_Function::RELU));
    layers->push_back(Layer(3, Activation_Function::RELU));
    Neural_Network model("Test Model", std::move(layers));
    for(int layer = 1; layer < 3; layer++){
        for(int i = 0; i < 3; i++){
            for(int j = 0; j < 3; j++){
                model.set_weight(layer, i, j, 1);
            }
        }
    }

    std::unique_ptr<Dataset> data = std::make_unique<Dataset>(Dataset({{1, 1, 1}, {2, 2, 2}, {4, 4, 4}}, {"a", "b", "c"} ,{{1, 1, 1}, {2, 2, 2}, {4, 4, 4}}));
    model.train_network(std::move(data), 0.1, 10, 0.0);

    std::vector<float> res = model.inference({1, 1, 1});
    for(int i = 0; i < res.size(); i++){
        std::cout << res[i] << std::endl;
    }
}
