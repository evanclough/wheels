#include <vector>
#include <memory>
#include <iostream>
#include "Linear_Regression.h"

    //constructors
    //default constructor defaults to input dim of 1, parameters set to 0, and no input/output data.
    Linear_Regression::Linear_Regression() {
        this->input_dim = 1;
        this->parameters = std::make_unique<std::vector<float>>(input_dim + 1, 0);
        this->input_data = std::make_unique<std::vector<std::vector<float>>>();
        this->output_data = std::make_unique<std::vector<float>>();
        this->training_data_size = 0;
    }

    //this constructor takes some initial input/output data, but sets parameters to 0 
    Linear_Regression::Linear_Regression(std::vector<std::vector<float>> initial_input_data, std::vector<float> initial_output_data) {
        this->input_dim = initial_input_data[0].size();
        this->parameters = std::make_unique<std::vector<float>>(input_dim + 1, 0);
        this->input_data = std::make_unique<std::vector<std::vector<float>>>(initial_input_data);
        this->output_data = std::make_unique<std::vector<float>>(initial_output_data);
        this->training_data_size = this->output_data->size();
    }

    //this constructor takes some initial input/output data, and some intiial parameters
    Linear_Regression::Linear_Regression(std::vector<std::vector<float>> initial_input_data, std::vector<float> initial_output_data, std::vector<float> initial_parameters){
        this->input_dim = initial_input_data[0].size();
        this->parameters = std::make_unique<std::vector<float>>(initial_parameters);
        this->input_data = std::make_unique<std::vector<std::vector<float>>>(initial_input_data);
        this->output_data = std::make_unique<std::vector<float>>(initial_output_data);
        this->training_data_size = this->output_data->size();
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

    //add input data to dataset
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
            this->input_data->push_back(input_data[i]);
            this->output_data->push_back(output_data[i]);
        }
        this->training_data_size += output_data.size();
    }

    // runs mean squared error on the current data set with current parameters
    float Linear_Regression::run_MSE() {
        float loss_accum = 0;
        std::vector<float> current_inferences = {};

        //gather current inferences
        for(std::vector<float> input_datum : *(this->input_data)){
            current_inferences.push_back(this->inference(input_datum));
        }
        
        //run through and accumulate MSE loss
        for(int i = 0; i < this->training_data_size; i++){
            float squared_loss = (current_inferences[i] - this->output_data->at(i)) * (current_inferences[i] - this->output_data->at(i));
            loss_accum += squared_loss;
        }

        //return mean
        return loss_accum / this->training_data_size;
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
                predicted += this->parameters->at(j) * this->input_data->at(i)[j - 1];
            }
            predicted_values.push_back(predicted);
        }
        
        //iterate through parameters, adjust according to learning rate multipled by the partial derivative of the loss function
        for(int i = 0; i < input_dim + 1; i++){
            float loss_accum = 0;
            for(int j = 0; j < this->training_data_size; j++){
                if(i != 0){//for weights
                    loss_accum += (predicted_values[j] - this->output_data->at(j)) * (this->input_data->at(j)[i - 1]);
                }else{//for bias
                    loss_accum += predicted_values[j] - this->output_data->at(j);
                }
            }

            (*(this->parameters))[i] -= learning_rate * (1.0f / this->training_data_size) * loss_accum; // decrease parameter by learnign rate multiplied by the partial derivative
        }

    }

    void Linear_Regression::train_model(float learning_rate, int epochs){
        std::cout << "Training model with learning_rate = " << learning_rate << " for " << epochs << " epochs..." << std::endl << std::endl;
        for(int i = 0; i < epochs; i++){
            std::cout << "Epoch " << i << ": " << std::endl;

            //run gradient descent with given learning rate 

            std::vector<float> old_params(*(this->parameters));
            this->gradient_descent(learning_rate);
            std::vector<float> new_params(*(this->parameters));
            
            //print changes in parameters for each epoch
            std::cout << "Bias: " << old_params[0] << " => "  << new_params[0] << std::endl;
            for(int i = 1; i < input_dim + 1; i++){
                std::cout << "Param " << i << ": " << old_params[i] << " => " << new_params[i] << std::endl;
            }
            std::cout << "Loss: " << this->run_MSE() << std::endl;
            std::cout << std::endl;
        }
    }

    //prints params of model
    void Linear_Regression::print_params(){
        //first print bias
        std::cout << "Bias: " << this->parameters->at(0) << std::endl;
        //then print parameters
        for(int i = 1; i < input_dim + 1; i++){
            std::cout << "Param " << i << ": " << this->parameters->at(i) << std::endl;
        }
        
    }