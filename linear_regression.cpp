#include <vector>
#include <iostream>
#include <string>
#include <stdexcept>
#include <random>

#include "Linear_Regression.h"


    //constructors
    //default constructor, params default to 0
    // @param model_name: name of the model. pass in empty string for it to default to 'model'.
    // @param num_features: number of features in model. must be positive integer.
    // @param feature_names: names of differnet features. leave blank to just default to indices.
    // @param training_data: Dataset object to be used as training data
    // @param test_data: Dataset object to be used as test data
    Linear_Regression::Linear_Regression(std::string model_name, int num_features, std::unique_ptr<Dataset> training_data, std::unique_ptr<Dataset> test_data){
        //set model name, default to just model if empty name passed in
        this->model_name = model_name == "" ? "Model" : model_name;

        //set num_features, throw error if it's negative/zero.
        if(num_features < 1){
            throw std::invalid_argument("num_features must be positive.");
        }
        this->num_features = num_features;

        //initialize datasets, validation to null until we need it
        this->training_data = std::move(training_data);
        this->test_data = std::move(test_data);
        //initalize parameters to zero
        std::vector<float> params_temp(num_features + 1, 0);
        this->parameters = std::make_unique<std::vector<float>>(params_temp);
    }

    //params constructor, params are passed in
    // @param model_name: name of the model. pass in empty string for it to default to 'model'.
    // @param num_features: number of features in model. must be positive integer.
    // @param feature_names: names of differnet features. leave blank to just default to indices.
    // @param training_data: Dataset object to be used as training data
    // @param test_data: Dataset object to be used as test data
    // @param initial_parameteres: initial parameters for model to use. bias is first element
    Linear_Regression::Linear_Regression(std::string model_name, int num_features, std::unique_ptr<Dataset> training_data, std::unique_ptr<Dataset> test_data, std::vector<float> initial_parameters){
        //set model name, default to just model if empty name passed in
        this->model_name = model_name == "" ? "Model" : model_name;

        //set num_features, throw error if it's negative/zero.
        if(num_features < 1){
            throw std::invalid_argument("num_features must be positive.");
        }
        this->num_features = num_features;

        //initialize datasets, validation to null until we need it
        this->training_data = std::move(training_data);
        this->test_data = std::move(test_data);
        //initalize parameters according to passed list, throw error if wrong size, remember to account for bias
        if(initial_parameters.size() != this->num_features + 1){
            throw std::invalid_argument("Must pass in proper number of initial parameters");
        }
        this->parameters = std::make_unique<std::vector<float>>(initial_parameters);
    }


    
    //makes inference on input with the current parameters
    float Linear_Regression::inference(std::vector<float> input) {
        //check if input is correct size, throw error if not.
        if(input.size() != this->num_features){
            throw std::invalid_argument("input array is of wrong dimension.");
        }

        //start with just bias
        float accum = this->parameters->at(0);
        for(int i = 1; i < this->num_features + 1; i++){
            accum += this->parameters->at(i) * input[i - 1];
        }
        return accum;
    }

    // runs mean squared error on the given data set with current parameters
    float Linear_Regression::run_MSE(Dataset_Type ds) {
        //set dataset variable 
        std::vector<std::vector<float>> feature_data;
        std::vector<float> label_data;
        int dataset_size;
        switch (ds) {
            case Dataset_Type::TRAINING:
                feature_data = this->training_data->get_feature_data();
                label_data = this->training_data->get_label_data();
                dataset_size = this->training_data->get_dataset_size();
            break;
            case Dataset_Type::TEST:
                feature_data = this->test_data->get_feature_data();
                label_data = this->test_data->get_label_data();
                dataset_size = this->test_data->get_dataset_size();
            break;
            case Dataset_Type::VALIDATION:
                feature_data = this->validation_data->get_feature_data();
                label_data = this->validation_data->get_label_data();
                dataset_size = this->validation_data->get_dataset_size();
            break;
        }
        
        //quick check to see if theres any data in dataset, throw error if not
        if(dataset_size == 0){
            throw std::invalid_argument("Dataset passed into MSE must not have size of zero.");
        }

        float loss_accum = 0;
        std::vector<float> current_inferences = {};
        //gather current inferences
        for(std::vector<float> feature_set : feature_data){
            current_inferences.push_back(this->inference(feature_set));
        }
        
        //run through and accumulate MSE loss
        for(int i = 0; i < dataset_size; i++){
            float squared_loss = (current_inferences[i] - label_data[i]) * (current_inferences[i] - label_data[i]);
            loss_accum += squared_loss;
        }
        
        //return mean
        return loss_accum / dataset_size;
    }

    

    // runs iteration of gradient descent using MSE as cost function given learning rate and current input data
    void Linear_Regression::gradient_descent(float learning_rate){
        //start wtih batch size of just one
        //for each feature-label pair we currently have in the data set, get the predicted value
        std::vector<float> predicted_values;
        for(int i = 0; i < this->training_data->get_dataset_size(); i++){
            //calculate current predicted value for given input, start with just bias
            float predicted = this->parameters->at(0);
            for(int j = 1; j < this->num_features + 1; j++){
                predicted += this->parameters->at(j) * this->training_data->get_feature_data()[i][j - 1];
            }
            predicted_values.push_back(predicted);
        }
        
        //iterate through parameters, adjust according to learning rate multipled by the partial derivative of the loss function
        for(int i = 0; i < this->num_features + 1; i++){
            float loss_accum = 0;
            for(int j = 0; j < this->training_data->get_dataset_size(); j++){
                if(i != 0){//for weights
                    loss_accum += (predicted_values[j] - this->training_data->get_label_data()[j]) * this->training_data->get_feature_data()[j][i - 1];
                }else{//for bias
                    loss_accum += predicted_values[j] - this->training_data->get_label_data()[j];
                }
            }

            (*(this->parameters))[i] -= learning_rate * (1.0f / this->training_data->get_dataset_size()) * loss_accum; // decrease parameter by learnign rate multiplied by the partial derivative
        }

    }

    void Linear_Regression::train_model(float learning_rate, int epochs, float validation_split){
        //first create training/validation split for this training run by picking randomly from training set
        int val_set_size = validation_split * this->training_data->get_dataset_size();

        //shuffle training data before val split
        this->training_data->shuffle_dataset();

        //pull data to create validation dataset with
        std::vector<std::vector<float>> validation_feature_data(this->training_data->get_feature_data().end() - val_set_size, this->training_data->get_feature_data().end());
        std::vector<float> validation_label_data(this->training_data->get_label_data().end() - val_set_size, this->training_data->get_label_data().end());
        this->validation_data = std::make_unique<Dataset>(Dataset(validation_feature_data, {}, validation_label_data));

        //remove validation data from training set
        for(int i = this->training_data->get_dataset_size() - 1; i >= this->training_data->get_dataset_size() - this->validation_data->get_dataset_size(); i--){
            this->training_data->remove_data_pair(i);
        }

        std::cout << "Training " << this->model_name << " with learning_rate = " << learning_rate << " for " << epochs << " epochs..." << std::endl << std::endl;
        for(int i = 0; i < epochs; i++){
            //first shuffle training data
            this->training_data->shuffle_dataset();

            //print epoch number
            std::cout << "Epoch " << i << ": " << std::endl;

            //run gradient descent with given learning rate 
            //log old and new params to print change
            std::vector<float> old_params(*(this->parameters));
            this->gradient_descent(learning_rate);
            std::vector<float> new_params(*(this->parameters));
            
            //print changes in parameters for each epoch
            std::cout << "Bias: " << old_params[0] << " => "  << new_params[0] << std::endl;
            for(int i = 1; i < this->num_features + 1; i++){
                std::cout << this->training_data->get_feature_names()[i - 1] << ": " << old_params[i] << " => " << new_params[i] << std::endl;
            }
            //print training loss and validation loss
            std::cout << "Training Loss: " << this->run_MSE(Dataset_Type::TRAINING) << std::endl;
            std::cout << "Validation Loss: " << this->run_MSE(Dataset_Type::VALIDATION) << std::endl;
            std::cout << std::endl;
        }

        //put validation data back into training and wipe it
        this->training_data->add_data_pairs(this->validation_data->get_feature_data(), this->validation_data->get_label_data());
        this->validation_data.reset();
    }

    //test model on current test data
    void Linear_Regression::test_model(){
        std::cout << "Testing " << this->model_name << "..." << std::endl;
        float test_loss = this->run_MSE(Dataset_Type::TEST);
        std::cout << "Test Loss: " << test_loss << std::endl;
    }

    //prints params of model
    void Linear_Regression::print_params(){
        //first print bias
        std::cout << "Bias: " << this->parameters->at(0) << std::endl;
        //then print parameters
        for(int i = 1; i < num_features + 1; i++){
            std::cout << this->training_data->get_feature_names()[i - 1] << ": " << this->parameters->at(i) << std::endl;
        }
    }