#include <vector>
#include <memory>
#include <string>

class Linear_Regression {
    private:
        int input_dim, training_data_size;
        std::unique_ptr<std::vector<float>> parameters;
        std::unique_ptr<std::vector<std::vector<float>>> input_data;
        std::unique_ptr<std::vector<float>> output_data;
        std::unique_ptr<std::vector<std::string>> param_names; // if left empty, no names

    public:
        //constructors
        Linear_Regression();
        Linear_Regression(std::vector<std::vector<float>> initial_input_data, std::vector<float> initial_output_data);
        Linear_Regression(std::vector<std::vector<float>> initial_input_data, std::vector<float> initial_output_data, std::vector<std::string> param_names);
        Linear_Regression(std::vector<std::vector<float>> initial_input_data, std::vector<float> initial_output_data, std::vector<float> initial_parameters);    

        //assorted utility functions

        //makes inference on input with the current parameters
        float inference(std::vector<float> input);

        //add training data to dataset
        void add_training_data(std::vector<std::vector<float>> input_data, std::vector<float> output_data);

        // runs mean squared error on the current data set with current parameters
        float run_MSE();
        
        // runs iteration of gradient descent using MSE as cost function given learning rate and current input data
        void gradient_descent(float learning_rate);
        
        //trains model via gradient descent, given a number of epochs and a learning rate
        void train_model(float learning_rate, int epochs);

        //prints parameters
        void print_params();
};