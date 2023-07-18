#include <vector>
#include <memory>
#include <iostream>
#include <string>
#include <stdexcept>
#include <algorithm>
#include <random>

#include "Linear_Regression.h"

    //constructors
    //basic constructor only takes numbero of parameters
    Linear_Regression::Linear_Regression(int num_params) {
        this->input_dim = num_params;
        this->model_name = "model";
        this->parameters = std::make_unique<std::vector<float>>(input_dim + 1, 0);
        this->training_input_data = std::make_unique<std::vector<std::vector<float>>>();
        this->training_output_data = std::make_unique<std::vector<float>>();
        this->test_input_data = std::make_unique<std::vector<std::vector<float>>>();
        this->test_output_data = std::make_unique<std::vector<float>>();
        this->validation_input_data = std::make_unique<std::vector<std::vector<float>>>();
        this->validation_output_data = std::make_unique<std::vector<float>>();
        this->training_data_size = 0;
        this->param_names = std::make_unique<std::vector<std::string>>();
    }

    //this constructor takes some initial input/output data, but sets parameters to 0 
    Linear_Regression::Linear_Regression(std::string model_name, std::vector<std::vector<float>> initial_training_input_data, std::vector<float> initial_training_output_data) {
        //check of inputted data to make sure dimensions match up
        if(initial_training_input_data.size() != initial_training_output_data.size()){
            throw std::invalid_argument("Input data size does not match output data size.");
        }

        //check if input data all same size
        int size_check = -1;
        for(int i = 0; i < initial_training_input_data.size(); i++){
            if(initial_training_input_data[i].size() != size_check && size_check != -1){
                throw std::invalid_argument("All input data must be of same dimension");
            }
            size_check = initial_training_input_data[i].size();
        }
        

        //once all good set attributes
        this->model_name = model_name == "" ? "Model" : model_name;
        this->input_dim = initial_training_input_data[0].size();
        this->parameters = std::make_unique<std::vector<float>>(input_dim + 1, 0);
        this->training_input_data = std::make_unique<std::vector<std::vector<float>>>(initial_training_input_data);
        this->training_output_data = std::make_unique<std::vector<float>>(initial_training_output_data);
        this->test_input_data = std::make_unique<std::vector<std::vector<float>>>();
        this->test_output_data = std::make_unique<std::vector<float>>();
        this->validation_input_data = std::make_unique<std::vector<std::vector<float>>>();
        this->validation_output_data = std::make_unique<std::vector<float>>();
        this->training_data_size = this->training_output_data->size();
        this->param_names = std::make_unique<std::vector<std::string>>();
    }

    //constructor same as above but takes in param_names vector
    Linear_Regression::Linear_Regression(std::string model_name, std::vector<std::vector<float>> initial_training_input_data, std::vector<float> initial_training_output_data, std::vector<std::string> param_names){
        //to use this constructor you must pass in some example data
        if(initial_training_input_data.size() == 0){
            throw std::invalid_argument("Initial training data vector must not be empty.");
        }
        
        //check of inputted data to make sure dimensions match up
        if(initial_training_input_data.size() != initial_training_output_data.size()){
            throw std::invalid_argument("Input data size does not match output data size.");
        }

        //check if input data all same size
        int size_check = -1;
        for(int i = 0; i < initial_training_input_data.size(); i++){
            if(initial_training_input_data[i].size() != size_check && size_check != -1){
                throw std::invalid_argument("All input data must be of same dimension");
            }
            size_check = initial_training_input_data[i].size();
        }
        
        //check if param names is appropriate size
        if(param_names.size() != initial_training_input_data[0].size()){
            throw std::invalid_argument("Number of parameter names passed in does not match number of parameters.");
        }

        this->model_name = model_name == "" ? "Model" : model_name;
        this->input_dim = initial_training_input_data[0].size();
        this->parameters = std::make_unique<std::vector<float>>(input_dim + 1, 0);
        this->training_input_data = std::make_unique<std::vector<std::vector<float>>>(initial_training_input_data);
        this->training_output_data = std::make_unique<std::vector<float>>(initial_training_output_data);
        this->training_data_size = this->training_output_data->size();
        this->test_input_data = std::make_unique<std::vector<std::vector<float>>>();
        this->test_output_data = std::make_unique<std::vector<float>>();
        this->validation_input_data = std::make_unique<std::vector<std::vector<float>>>();
        this->validation_output_data = std::make_unique<std::vector<float>>();
        this->param_names = std::make_unique<std::vector<std::string>>(param_names);
    }

    // this constructor takes initial training data and initial test data and param names
    Linear_Regression::Linear_Regression(std::string model_name, std::vector<std::vector<float>> initial_training_input_data, std::vector<float> initial_training_output_data, std::vector<std::vector<float>> initial_test_input_data, std::vector<float> initial_test_output_data, std::vector<std::string> param_names){
        //to use this constructor you must pass in some example training and test data
        if(initial_training_input_data.size() == 0){
            throw std::invalid_argument("Initial training data vector must not be empty.");
        }
        if(initial_test_input_data.size() == 0){
            throw std::invalid_argument("Initial test data vector must not be empty.");
        }
        
        //check of training and test input data to make sure dimensions match up
        if(initial_training_input_data.size() != initial_training_output_data.size()){
            throw std::invalid_argument("Training input data size does not match training output data size.");
        }
        if(initial_test_input_data.size() != initial_test_output_data.size()){
            throw std::invalid_argument("Test input data size does not match test output data size.");
        }

        //check if training and test input data all same size
        int size_check = -1;
        for(int i = 0; i < initial_training_input_data.size(); i++){
            if(initial_training_input_data[i].size() != size_check && size_check != -1){
                throw std::invalid_argument("All input data must be of same dimension");
            }
            size_check = initial_training_input_data[i].size();
        }
        size_check = -1;
        for(int i = 0; i < initial_test_input_data.size(); i++){
            if(initial_test_input_data[i].size() != size_check && size_check != -1){
                throw std::invalid_argument("All input data must be of same dimension");
            }
            size_check = initial_test_input_data[i].size();
        }
        
        //check if param names is appropriate size
        if(param_names.size() != initial_training_input_data[0].size()){
            throw std::invalid_argument("Number of parameter names passed in does not match number of parameters.");
        }

        this->model_name = model_name == "" ? "Model" : model_name;
        this->input_dim = initial_training_input_data[0].size();
        this->parameters = std::make_unique<std::vector<float>>(input_dim + 1, 0);
        this->training_input_data = std::make_unique<std::vector<std::vector<float>>>(initial_training_input_data);
        this->training_output_data = std::make_unique<std::vector<float>>(initial_training_output_data);
        this->training_data_size = this->training_output_data->size();
        this->test_input_data = std::make_unique<std::vector<std::vector<float>>>(initial_test_input_data);
        this->test_output_data = std::make_unique<std::vector<float>>(initial_test_output_data);
        this->test_data_size = this->test_output_data->size();
        this->validation_input_data = std::make_unique<std::vector<std::vector<float>>>();
        this->validation_output_data = std::make_unique<std::vector<float>>();
        this->param_names = std::make_unique<std::vector<std::string>>(param_names);
    }

    //this constructor takes some initial input/output data, and some intiial parameters
    Linear_Regression::Linear_Regression(std::string model_name, std::vector<std::vector<float>> initial_training_input_data, std::vector<float> initial_training_output_data, std::vector<float> initial_parameters){
        //to use this constructor you must pass in some example data
        if(initial_training_input_data.size() == 0){
            throw std::invalid_argument("Initial training data vector must not be empty.");
        }
        
        //check of inputted data to make sure dimensions match up
        if(initial_training_input_data.size() != initial_training_output_data.size()){
            throw std::invalid_argument("Input data size does not match output data size.");
        }

        //check if input data all same size
        int size_check = -1;
        for(int i = 0; i < initial_training_input_data.size(); i++){
            if(initial_training_input_data[i].size() != size_check && size_check != -1){
                throw std::invalid_argument("All input data must be of same dimension");
            }
            size_check = initial_training_input_data[i].size();
        }
        
        //check if initial param array is appropriate size
        if(initial_parameters.size() != initial_training_input_data[0].size()){
            throw std::invalid_argument("Number of parameters passed in must be equal to number of parameters indicated by input data size.");
        }

        this->model_name = model_name == "" ? "Model" : model_name;
        this->input_dim = initial_training_input_data[0].size();
        this->parameters = std::make_unique<std::vector<float>>(initial_parameters);
        this->training_input_data = std::make_unique<std::vector<std::vector<float>>>(initial_training_input_data);
        this->training_output_data = std::make_unique<std::vector<float>>(initial_training_output_data);
        this->test_input_data = std::make_unique<std::vector<std::vector<float>>>();
        this->test_output_data = std::make_unique<std::vector<float>>();
        this->validation_input_data = std::make_unique<std::vector<std::vector<float>>>();
        this->validation_output_data = std::make_unique<std::vector<float>>();
        this->training_data_size = this->training_output_data->size();
        this->param_names = std::make_unique<std::vector<std::string>>();
    }

    //makes inference on input with the current parameters
    float Linear_Regression::inference(std::vector<float> input) {
        //check if input is correct size, return null and tell them if not.
        if(input.size() != input_dim){
            std::cout << "Error: expected input data dimension: " << input_dim << ". Received input data dimension: " << input.size();
            return -1.0;
        }

        float accum = this->parameters->at(0);
        for(int i = 0; i < input_dim; i++){
            accum += this->parameters->at(i + 1) * input[i];
        }
        return accum;
    }

    //add training data to dataset
    void Linear_Regression::add_training_data(std::vector<std::vector<float>> input_data, std::vector<float> output_data) {
        //input dimension size check
        for(std::vector<float> datum : input_data){
            if(datum.size() != input_dim){
                std::cout << "Error: expected input data dimension: " << input_dim << ". Received input data dimension: " << input_data.size() << "\n";
                return;
            }
        }

        //input/output data size match check
        if(input_data.size() != output_data.size()){
            std::cout << "Error: input size doesn't match output size." << "\n";
        }

        //add data
        for(int i = 0; i < input_data.size(); i++){
            this->training_input_data->push_back(input_data[i]);
            this->training_output_data->push_back(output_data[i]);
        }
        this->training_data_size += output_data.size();
    }

    //add test data to dataset
    void Linear_Regression::add_test_data(std::vector<std::vector<float>> input_data, std::vector<float> output_data) {
        //input dimension size check
        for(std::vector<float> datum : input_data){
            if(datum.size() != input_dim){
                std::cout << "Error: expected input data dimension: " << input_dim << ". Received input data dimension: " << input_data.size() << "\n";
                return;
            }
        }

        //input/output data size match check
        if(input_data.size() != output_data.size()){
            std::cout << "Error: input size doesn't match output size." << "\n";
        }

        //add data
        for(int i = 0; i < input_data.size(); i++){
            this->test_input_data->push_back(input_data[i]);
            this->test_output_data->push_back(output_data[i]);
        }
        this->test_data_size += output_data.size();
    }

    // runs mean squared error on the given data set with current parameters
    float Linear_Regression::run_MSE(Dataset ds) {
        std::vector<std::vector<float>> feature_data;
        std::vector<float> label_data;
        int data_size;

        switch (ds) {
            case Dataset::TRAINING:
                feature_data = *(this->training_input_data);
                label_data = *(this->training_output_data);
                data_size = this->training_data_size;
            break;
            case Dataset::TEST:
                feature_data = *(this->test_input_data);
                label_data = *(this->test_output_data);
                data_size = this->test_data_size;
            break;
            case Dataset::VALIDATION:
                feature_data = *(this->validation_input_data);
                label_data = *(this->validation_output_data);
                data_size = this->validation_data_size;
            break;
        }
        
        //quick check to see if theres any data in dataset, throw error if not
        if(data_size == 0){
            throw std::invalid_argument("Dataset passed into MSE must not have size of zero.");
        }


        float loss_accum = 0;
        std::vector<float> current_inferences = {};
        //gather current inferences
        for(std::vector<float> input_datum : feature_data){
            current_inferences.push_back(this->inference(input_datum));
        }
        
        //run through and accumulate MSE loss
        for(int i = 0; i < data_size; i++){
            float squared_loss = (current_inferences[i] - label_data[i]) * (current_inferences[i] - label_data[i]);
            loss_accum += squared_loss;
        }
        
        //return mean
        return loss_accum / data_size;
    }

    

    // runs iteration of gradient descent using MSE as cost function given learning rate and current input data
    void Linear_Regression::gradient_descent(float learning_rate){
        //start wtih batch size of just one
        //for each feature-label pair we currently have in the data set, get the predicted value
        std::vector<float> predicted_values;
        for(int i = 0; i < this->training_data_size; i++){
            //calculate current predicted value for given input, start with just bias
            float predicted = this->parameters->at(0);
            for(int j = 1; j < this->input_dim + 1; j++){
                predicted += this->parameters->at(j) * this->training_input_data->at(i)[j - 1];
            }
            predicted_values.push_back(predicted);
        }
        
        //iterate through parameters, adjust according to learning rate multipled by the partial derivative of the loss function
        for(int i = 0; i < input_dim + 1; i++){
            float loss_accum = 0;
            for(int j = 0; j < this->training_data_size; j++){
                if(i != 0){//for weights
                    loss_accum += (predicted_values[j] - this->training_output_data->at(j)) * (this->training_input_data->at(j)[i - 1]);
                }else{//for bias
                    loss_accum += predicted_values[j] - this->training_output_data->at(j);
                }
            }

            (*(this->parameters))[i] -= learning_rate * (1.0f / this->training_data_size) * loss_accum; // decrease parameter by learnign rate multiplied by the partial derivative
        }

    }

    void Linear_Regression::train_model(float learning_rate, int epochs, float validation_split){
        //first create training/validation split for this training run by picking randomly from training set
        int val_set_size = validation_split * this->training_data_size;
        std::cout << "val set size" << val_set_size << std::endl;
        this->validation_data_size = val_set_size;
        //stuff for generating random index to pull
        std::random_device rd;
        std::mt19937 gen(rd());

        //shuffle training data for val split pull
        std::shuffle(this->training_input_data->begin(), this->training_input_data->end(), gen);
        std::shuffle(this->training_output_data->begin(), this->training_output_data->end(), gen);

        for(int i = 0; i < val_set_size; i++){
            //copy over shuffled first few elements
            this->validation_input_data->push_back(this->training_input_data->at(i));
            this->validation_output_data->push_back(this->training_output_data->at(i));
        }
        //remove from training set
        this->training_input_data->erase(this->training_input_data->begin(), this->training_input_data->begin() + val_set_size);
        this->training_output_data->erase(this->training_output_data->begin(), this->training_output_data->begin() + val_set_size);
        this->training_data_size -= val_set_size;

        std::cout << "Training " << this->model_name << " with learning_rate = " << learning_rate << " for " << epochs << " epochs..." << std::endl << std::endl;
        for(int i = 0; i < epochs; i++){
            //first shuffle training data
            std::shuffle(this->training_input_data->begin(), this->training_input_data->end(), gen);
            std::shuffle(this->training_output_data->begin(), this->training_output_data->end(), gen);

            //print epoch number
            std::cout << "Epoch " << i << ": " << std::endl;

            //run gradient descent with given learning rate 
            //log old and new params to print change
            std::vector<float> old_params(*(this->parameters));
            this->gradient_descent(learning_rate);
            std::vector<float> new_params(*(this->parameters));
            
            //print changes in parameters for each epoch
            std::cout << "Bias: " << old_params[0] << " => "  << new_params[0] << std::endl;
            for(int i = 1; i < input_dim + 1; i++){
                if(this->param_names->size() == 0){
                    std::cout << "Param " << i << ": " << old_params[i] << " => " << new_params[i] << std::endl;
                }else {
                    std::cout << this->param_names->at(i - 1) << ": " << old_params[i] << " => " << new_params[i] << std::endl;
                }
            }
            //print training loss and validation loss
            std::cout << "Training Loss: " << this->run_MSE(Dataset::TRAINING) << std::endl;
            std::cout << "Validation Loss: " << this->run_MSE(Dataset::VALIDATION) << std::endl;
            std::cout << std::endl;
        }

        //put validation data back into training and wipe it
        for(int i = 0; i < val_set_size; i++){
            this->training_input_data->push_back(this->validation_input_data->at(i));
            this->training_output_data->push_back(this->validation_output_data->at(i));
        }
        this->training_data_size += val_set_size;
        this->validation_input_data->clear();
        this->validation_output_data->clear();
        this->validation_data_size = 0;
    }

    //test model on current test data
    void Linear_Regression::test_model(){
        std::cout << "Testing " << this->model_name << "..." << std::endl;
        float test_loss = this->run_MSE(Dataset::TEST);
        std::cout << "Test Loss: " << test_loss << std::endl;
    }

    //prints params of model
    void Linear_Regression::print_params(){
        //first print bias
        std::cout << "Bias: " << this->parameters->at(0) << std::endl;
        //then print parameters
        for(int i = 1; i < input_dim + 1; i++){
            if(this->param_names->size() == 0){
                std::cout << "Param " << i << ": " << this->parameters->at(i) << std::endl;
            }else {
                std::cout << this->param_names->at(i - 1) << ": " << this->parameters->at(i) << std::endl;
            }
            
        }
        
    }