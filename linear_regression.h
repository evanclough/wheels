#include <vector>
#include <string>
#include <memory>
#include "Dataset.h"

enum class Dataset_Type {
    TRAINING,
    TEST,
    VALIDATION
};

class Linear_Regression {
    private:
        int num_features;
        std::string model_name;
        std::unique_ptr<std::vector<float>> parameters;
        std::unique_ptr<Dataset> validation_data;
    public:
        //expose these publicly to allow esiting of datasets
        std::unique_ptr<Dataset> training_data;
        std::unique_ptr<Dataset> test_data;

        //constructors. one with initial contstrucors and one without
        Linear_Regression(std::string model_name, int num_features, std::unique_ptr<Dataset> training_data, std::unique_ptr<Dataset> test_data);
        Linear_Regression(std::string model_name, int num_features, std::unique_ptr<Dataset> training_data, std::unique_ptr<Dataset> test_data, std::vector<float> initial_parameters);
        //assorted utility functions

        //makes inference on input with the current parameters
        float inference(std::vector<float> input);

        // runs mean squared error on given data set with current parameters
        float run_MSE(Dataset_Type ds);
        
        // runs iteration of gradient descent using MSE as cost function given learning rate and current input data
        void gradient_descent(float learning_rate);
        
        //trains model via gradient descent, given a number of epochs, a validation split size, and a learning rate
        void train_model(float learning_rate, int epochs, float validation_split);

        //tests model against provided test data
        void test_model();

        //prints parameters
        void print_params();
};