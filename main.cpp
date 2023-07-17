#include <iostream>
#include "Linear_Regression.h"

int main() {
    Linear_Regression test_model("test model", {{1, 1}, {2, 2}, {3, 3}}, {5, 9, 13}, {{1, 1}, {2, 2}, {3, 3}}, {5, 9, 13}, {"apples", "bananas"});
    test_model.train_model(0.1, 10, 0.2);
    test_model.test_model();
}