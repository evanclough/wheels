#include <iostream>
#include "Linear_Regression.h"

int main() {
    Linear_Regression model("My Test Model", 3, std::make_unique<Dataset>(Dataset({{1, 1, 1}, {2, 2, 2}, {3, 3, 3}, {4, 4, 4}, {5, 5, 5}, {6, 6, 6}}, {"apples", "bananas", "oranges"}, {1, 2, 3, 4, 5, 6})), std::make_unique<Dataset>(Dataset({{1, 1, 1}}, {}, {1})));
    model.train_model(0.1, 10, 0.2);
}