#include "Neural_Network.h"
#include <stdexcept>

//basic constructor takes a model name and a layer array to create network
Neural_Network::Neural_Network(std::string model_name, std::unique_ptr<std::vector<Layer>> layers){
    //check if layers array has at least two layers, throw error otherwise.
    if(layers->size() < 2){
        throw std::invalid_argument("Layers array must include at least two layers.");
    }
    this->layers = std::move(layers);
    this->num_layers = this->layers->size();
    this->num_features = this->layers->at(0).get_size();

    //initialize weights and biases in layers to 0
    for(int i = 1; i < this->num_layers; i++){
        this->layers->at(i).set_default(this->layers->at(i - 1).get_size());
    }
    this->model_name = model_name == "" ? "Model" : model_name;
}

//runs neural network with given input data and an activation function
std::vector<float> Neural_Network::inference(std::vector<float> input, Activation_Function activation){
    std::vector<float> temp = input;
    for(int i = 1; i < this->num_layers; i++){
        temp = this->layers->at(i).evaluate(temp, activation);
    }
    return temp;
}

//runs backpropogation on network given feature and label vectors
void Neural_Network::backprop(std::vector<float> feature, std::vector<float> labels, Activation_Function activation){
    ;
}

//trains network with a given training dataset, learning rate, number of epochs, and validation split
//pretty mucht he same as the linear regression training function with some small changes
void Neural_Network::train_network(Dataset training_data, float learning_rate, int epochs, float validation_split){
    //first set size of validation training set
    int val_set_size = validation_split * training_data.get_dataset_size();

    //shuffle training dataset before split to make sure it's random
    training_data.shuffle_dataset();

    //if there's an allocated validation split, create it
    Dataset* validation_data;
    if(val_set_size > 0){
        std::vector<std::vector<float>> validation_feature_data;
        std::vector<std::vector<float>> validation_label_data;
        for(int i = training_data.get_dataset_size() - val_set_size; i < training_data.get_dataset_size(); i++){
            validation_feature_data.push_back(training_data.get_feature_data()[i]);
            validation_label_data.push_back(training_data.get_label_data()[i]);
        }
        validation_data = new Dataset(validation_feature_data, {}, validation_label_data);
    }

    int initial_training_data_size = training_data.get_dataset_size();
    //remove validation data from training set
    for(int i = training_data.get_dataset_size() - 1; i >= initial_training_data_size - val_set_size; i--){
        training_data.remove_data_pair(i);
    }

    std::cout << "Training " << this->model_name << " with learning_rate = " << learning_rate << " for " << epochs << " epochs..." << std::endl << std::endl;
    for(int i = 0; i < epochs; i++){
        //shuffle training data before each epoch
        training_data.shuffle_dataset();

        //print epoch number
        std::cout << "Epoch " << i << ": " << std::endl;

        //run backpropagation with each feature/value pair in training dataset
        std::vector<std::vector<float>> feature_data = training_data.get_feature_data();
        std::vector<std::vector<float>> label_data = training_data.get_label_data();
        for(int j = 0; j < training_data.get_dataset_size(); j++){
            this->backprop(feature_data[j], label_data[j], Activation_Function::SIGMOID);
        }
    }
}

//tests network on given test data set and returns error
void Neural_Network::test_network(Dataset test_data){
    ;
}