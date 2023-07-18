#include <vector>
#include <memory>
#include <string>

enum class Dataset {
    TRAINING,
    TEST,
    VALIDATION
};

class Linear_Regression {
    private:
        int input_dim, training_data_size, test_data_size, validation_data_size;
        std::string model_name;
        std::unique_ptr<std::vector<float>> parameters;
        std::unique_ptr<std::vector<std::vector<float>>> training_input_data;
        std::unique_ptr<std::vector<float>> training_output_data;
        std::unique_ptr<std::vector<std::vector<float>>> test_input_data;
        std::unique_ptr<std::vector<float>> test_output_data;
        std::unique_ptr<std::vector<std::vector<float>>> validation_input_data;
        std::unique_ptr<std::vector<float>> validation_output_data;
        std::unique_ptr<std::vector<std::string>> param_names; // if left empty, no names

    public:
        //constructors
        Linear_Regression(int num_params);
        Linear_Regression(std::string model_name, std::vector<std::vector<float>> initial_training_input_data, std::vector<float> initial_training_output_data);
        Linear_Regression(std::string model_name, std::vector<std::vector<float>> initial_training_input_data, std::vector<float> initial_training_output_data, std::vector<std::string> param_names);
        Linear_Regression(std::string model_name, std::vector<std::vector<float>> initial_training_input_data, std::vector<float> initial_training_output_data, std::vector<std::vector<float>> initial_test_input_data, std::vector<float> initial_test_output_data, std::vector<std::string> param_names);
        Linear_Regression(std::string model_name, std::vector<std::vector<float>> initial_training_input_data, std::vector<float> initial_training_output_data, std::vector<float> initial_parameters);    

        //assorted utility functions

        //makes inference on input with the current parameters
        float inference(std::vector<float> input);

        //add training data to dataset
        void add_training_data(std::vector<std::vector<float>> input_data, std::vector<float> output_data);

        //add training data to dataset
        void add_test_data(std::vector<std::vector<float>> input_data, std::vector<float> output_data);

        // runs mean squared error on given data set with current parameters
        float run_MSE(Dataset ds);
        
        // runs iteration of gradient descent using MSE as cost function given learning rate and current input data
        void gradient_descent(float learning_rate);
        
        //trains model via gradient descent, given a number of epochs and a learning rate
        void train_model(float learning_rate, int epochs, float validation_split);

        //tests model against provided test data
        void test_model();

        //prints parameters
        void print_params();
};