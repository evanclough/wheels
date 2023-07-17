#include <iostream>
#include "Linear_Regression.h"

int main() {
    Linear_Regression test_model({{1, 1}, {2, 2}, {3, 3}}, {5, 9, 13});
    test_model.train_model(0.1, 10000);
    test_model.print_params();
}