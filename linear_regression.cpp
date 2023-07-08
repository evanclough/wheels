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
        if(input.size() != input_dim - 1){
            std::cout << "Error: expected input data dimension: " << input_dim - 1 << ". Received input data dimension: " << input.size();
            return -1.0;
        }

        float accum = this->parameters->at(0);
        for(int i = 1; i <= input_dim; i++){
            accum += this->parameters->at(i) * input[i];
        }
        return accum;
    }

    //add input data to dataset
    void Linear_Regression::add_training_data(std::vector<std::vector<float>> input_data, std::vector<float> output_data) {
        //input dimension size check
        for(std::vector<float> datum : input_data){
            if(datum.size() != input_dim - 1){
                std::cout << "Error: expected input data dimension: " << input_dim - 1 << ". Received input data dimension: " << input_data.size() << "\n";
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