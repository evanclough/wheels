#include <iostream>
#include "Linear_Regression.h"

int main() {
    Linear_Regression my_model = Linear_Regression(std::vector<float>({1.0, 2.0, 3.0}), std::vector<float>({2.0, 4.0, 6.0}), 2.0, 0.0);
    float loss = my_model.run_MSE();
    std::cout << loss;
}