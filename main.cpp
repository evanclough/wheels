#include <iostream>
#include "Linear_Regression.h"

int main() {
    std::unique_ptr<Dataset> training_ds = std::make_unique<Dataset>(Dataset("test.csv", {"apples", "bananas"}, "oranges"));
    std::unique_ptr<Dataset> test_ds = std::make_unique<Dataset>(Dataset(std::vector<std::vector<float>>({{1, 2}}), {},  std::vector<float>({3})));
    Linear_Regression model("My Test Model", 2, std::move(training_ds), std::move(test_ds));
    model.train_model(0.01, 100, 0.0);
}