#include "../Linear_Regression.h"

//demonstration of the basic functions of the linear regression class of the library.
//for now, you can compile it with g++ *.cpp ..\*.cpp

int main() {
    //first initialize our training dataset, we'll do it from the the my_data csv
    //we're going to want to capture apples and bananas as our feature columns, and oranges as our label column.
    std::unique_ptr<Dataset> training_ds = std::make_unique<Dataset>(Dataset("fruit_data.csv", {"apples", "bananas"}, {"oranges"}));

    //then initialize our test dataset, we're going to do it directly in code.
    std::unique_ptr<Dataset> test_ds = std::make_unique<Dataset>(Dataset(std::vector<std::vector<float>>({{1, 2}}), {"apples", "bananas"},  std::vector<std::vector<float>>({{5}})));

    //then we can initialize our model wtih both of our datasets
    Linear_Regression my_model("Fruits", 2, std::move(training_ds), std::move(test_ds));
    
    //then we can train our model with the train_model method. we're going to see how the number of apples and bananas someone has correlates with the number of oranges.
    //the train_model function takes in a learning rate, number of epochs to run, and validation split. 
    my_model.train_model(0.01, 100, 0.2);

    //we can then test our model against our test dataset to see how well it does.
    my_model.test_model();
}