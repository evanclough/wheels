# wheels
A C++ library with basic neural network and linear regression functionality from the ground up to teach myself the fundamentals. "Re-inventing the wheel". Most of the features are inspired by the Google Machine Learning Crash Course

## Usage

### Compilation: 
For now, you can just do 
``` g++ *.cpp -o output```

### Datasets
You can create a Dataset via vectors directly created in code, or from a specified CSV, with specified columns to use as features/labels. View the constructors for more info.

### Linear Regression
You can create a linear regression model with a specified model name, training dataset, and test dataset. You can train the model with the train_model function, with a specified learning rate, number of epochs, and validation split. You can test the model against the test dataset with the test_model.
