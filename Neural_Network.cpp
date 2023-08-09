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
std::vector<float> Neural_Network::inference(std::vector<float> input){
    std::vector<float> temp = input;
    for(int i = 1; i < this->num_layers; i++){
        temp = this->layers->at(i).evaluate(temp);
    }
    return temp;
}

//fetches activations of network on given input
std::vector<std::vector<float>> Neural_Network::activations(std::vector<float> input){
    std::vector<std::vector<float>> accum;
    std::vector<float> temp = input;
    for(int i = 1; i < this->num_layers; i++){
        accum.push_back(temp);
        temp = this->layers->at(i).evaluate(temp);
    }
    accum.push_back(temp);
    return accum;
}

//fetches unactivated layers
std::vector<std::vector<float>> Neural_Network::z_values(std::vector<float> input){
    std::vector<std::vector<float>> activations = this->activations(input);
    std::vector<std::vector<float>> accum = {input};
    for(int i = 1; i < activations.size(); i++){
        accum.push_back(this->layers->at(i).evaluate_without_activation(activations[i - 1]));
    }
    return accum;
}

//runs MSE loss on given dataset
std::vector<float> Neural_Network::run_MSE(std::unique_ptr<Dataset> data){
    std::vector<std::vector<float>> feature_data = data->get_feature_data();
    std::vector<std::vector<float>> label_data = data->get_label_data();
    std::vector<std::vector<float>> predicted;
    for(int i = 0; i < data->get_dataset_size(); i++){
        predicted.push_back(this->inference(feature_data[i]));
    }
    std::vector<float> loss_accum(data->get_num_labels(), 0);
    for(int i = 0; i < data->get_dataset_size(); i++){
        for(int j = 0; j < data->get_num_labels(); j++){
            loss_accum[j] += (predicted[i][j] - label_data[i][j]) * (predicted[i][j] - label_data[i][j]);
        }
    }
    for(int i = 0; i < data->get_num_labels(); i++){
        loss_accum[i] /= data->get_dataset_size();
    }
    return loss_accum;
}

//runs backpropogation on network given feature and label vectors and a learning rate
//naive implementation with complexity of o(L!)
void Neural_Network::backprop(std::vector<float> feature, std::vector<float> label, float learning_rate){
    //fetch activations and z values for whole network
    this->temp_activations = this->activations(feature);
    this->temp_z_values = this->z_values(feature);

    //for each weight and bias in network, subtract the learning rate times the derivative of the cost function, MSE, with respect to the given weight or bias.
    for(int i = this->layers->size() - 1; i >= 0; i--){
        for(int j = 0; j < this->layers->at(i).get_size(); j++){
            this->layers->at(i).set_bias(j, this->layers->at(i) .get_nodes()[j].bias - learning_rate * pd_bias(feature, label, i, j));
            for(int k = 0; k < this->layers->at(i).get_nodes()[j].weights.size(); j++){
                this->layers->at(i).set_weight(i, j, this->layers->at(i).get_nodes()[j].weights[k] - learning_rate * this->pd_weight(feature, label, i, j, k));
            }
        }
    }

    //reset temp activations and z values array
    this->temp_activations = {};
    this->temp_z_values = {};
}

//derivative of the current activation function of the network
float activation_derivative(float input, Activation_Function activation){
    switch(activation){
        case SIGMOID:
            return ((1 / (1 + std::pow(2.71828, -input)))) * (1 - ((1 / (1 + std::pow(2.71828, -input)))));
        break;
        case TANH:
            return 1 - (std::tanh(input) * std::tanh(input));
        break;
        case RELU:
            return input < 0 ? 0 : 1;
        break;
        default:
            return input;
    }
}

//calculates the partiaul derivative of a given weight with respect to the cost function. (naive implementation)
float Neural_Network::pd_weight(std::vector<float> feature, std::vector<float> label,int layer, int j, int k){
    
    //accumulator to be used throughout calculation
    float accum = 0;

    //temp array for storning values to be summed later
    std::vector<float> temp;
    for(int i = 0; i < label.size(); i++){
        temp.push_back(2 * (this->temp_activations[this->num_layers - 1][i] - label[i]) * this->activation_derivative(this->temp_z_values[this->num_layers - 1][i], this->layers->at(this->num_layers - 1).get_activation()) * pd_z_wrt_weight(layer, j, k, layer, i));
    }

    //sum up temp array whil emultiplying by final weights activation 
    for(int i = 0; i < temp.size(); i++){
        accum += temp[i];
    }

    //multiply by 1 over size of out put layer to finish calculation
    return accum * 1.0 / label.size();
}

//finds partial derivative of a given z value with respect to a given weight, recursive helpyer to pd_weight
float Neural_Network::pd_z_wrt_weight(int weight_layer, int weight_j, int weight_k, int z_layer, int z_j){
    //if weight is in z layer, just return the activation it's attached to
    if(weight_layer == z_layer){
        return this->temp_activations[z_layer - 1][weight_k];
    }else{
        //otherwise, return summation of weights times derivative of activations in previous layer
        float accum = 0;
        for(int i = 0; i < this->layers->at(z_layer - 1).get_size(); i++){
            accum += this->layers->at(z_layer).get_nodes()[z_j].weights[i] * this->activation_derivative(this->temp_z_values[z_layer - 1][i], this->layers->at(z_layer - 1).get_activation()) * this->pd_z_wrt_weight(weight_layer, weight_j, weight_k, z_layer - 1, i);
        }   
        return accum;
    }
}

float Neural_Network::pd_bias(std::vector<float> feature, std::vector<float> label,int layer, int j){
    //accumulator to be used throughout calculation
    float accum = 0;

    //temp array for storning values to be summed later
    std::vector<float> temp;
    for(int i = 0; i < label.size(); i++){
        temp.push_back(2 * (this->temp_activations[this->num_layers - 1][i] - label[i]) * this->activation_derivative(this->temp_z_values[this->num_layers - 1][i], this->layers->at(this->num_layers - 1).get_activation()) * pd_z_wrt_bias(layer, j, layer, i));
    }

    //sum up temp array whil emultiplying by final weights activation 
    for(int i = 0; i < temp.size(); i++){
        accum += temp[i];
    }

    //multiply by 1 over size of out put layer to finish calculation
    return accum * 1.0 / label.size();
}

//finds partial derivative of a given z value with respect to a given weight, recursive helpyer to pd_weight
float Neural_Network::pd_z_wrt_bias(int bias_layer, int bias_j, int z_layer, int z_j){
    //if weight is in z layer, just return the activation it's attached to
    if(bias_layer == z_layer){
        return 1;
    }else{
        //otherwise, return summation of weights times derivative of activations in previous layer
        float accum = 0;
        for(int i = 0; i < this->layers->at(z_layer - 1).get_size(); i++){
            accum += this->layers->at(z_layer).get_nodes()[z_j].weights[i] * this->activation_derivative(this->temp_z_values[z_layer - 1][i], this->layers->at(z_layer - 1).get_activation()) * this->pd_z_wrt_bias(bias_layer, bias_j, z_layer - 1, i);
        }   
        return accum;
    }
}

//trains network with a given training dataset, learning rate, number of epochs, and validation split
//pretty mucht he same as the linear regression training function with some small changes
void Neural_Network::train_network(std::unique_ptr<Dataset> training_data, float learning_rate, int epochs, float validation_split){
    //first set size of validation training set
    int val_set_size = validation_split * training_data->get_dataset_size();

    //shuffle training dataset before split to make sure it's random
    training_data->shuffle_dataset();

    //if there's an allocated validation split, create it
    std::unique_ptr<Dataset> validation_data;
    if(val_set_size > 0){
        std::vector<std::vector<float>> validation_feature_data;
        std::vector<std::vector<float>> validation_label_data;
        for(int i = training_data->get_dataset_size() - val_set_size; i < training_data->get_dataset_size(); i++){
            validation_feature_data.push_back(training_data->get_feature_data()[i]);
            validation_label_data.push_back(training_data->get_label_data()[i]);
        }
        validation_data = std::make_unique<Dataset>(Dataset(validation_feature_data, {}, validation_label_data));
    }

    int initial_training_data_size = training_data->get_dataset_size();
    //remove validation data from training set
    for(int i = training_data->get_dataset_size() - 1; i >= initial_training_data_size - val_set_size; i--){
        training_data->remove_data_pair(i);
    }

    std::cout << "Training " << this->model_name << " with learning_rate = " << learning_rate << " for " << epochs << " epochs..." << std::endl << std::endl;
    for(int i = 0; i < epochs; i++){
        //shuffle training data before each epoch
        training_data->shuffle_dataset();

        //print epoch number
        std::cout << "Epoch " << i << ": " << std::endl;

        //run backpropagation with each feature/value pair in training dataset
        std::vector<std::vector<float>> feature_data = training_data->get_feature_data();
        std::vector<std::vector<float>> label_data = training_data->get_label_data();
        for(int j = 0; j < training_data->get_dataset_size(); j++){
            this->backprop(feature_data[j], label_data[j], learning_rate);
        }
        //print loss
        std::cout << "Training Loss:";
        std::vector<float> training_loss = this->run_MSE(std::move(training_data));
        for(int j = 0; j < training_loss.size(); j++){
            std::cout << " " << training_loss[j] << " " ;
        }
        std::cout << std::endl;

        //same for validation if there is any
        std::cout << "Validation Loss:";
        std::vector<float> validation_loss = this->run_MSE(std::move(validation_data));
        for(int j = 0; j < validation_loss.size(); j++){
            std::cout << " " << validation_loss[j] << " " ;
        }
        std::cout << std::endl;
    }
}

//tests network on given test data set and returns error
void Neural_Network::test_network(std::unique_ptr<Dataset> test_data){
    std::cout << "Testing " << this->model_name << "..." << std::endl;
    std::vector<float> test_loss = this->run_MSE(std::move(test_data));
    std::cout << "Test Loss:";
    for(int i = 0; i < test_loss.size(); i++){
        std::cout << " " << test_loss[i] << " ";
    }
    std::cout << std::endl;
}

void Neural_Network::print_network(){
    //print input layer
    for(int i = 0; i < this->layers->at(0).get_size(); i++){
        std::cout << " 0 " << "\t";
    }
    std::cout << std::endl;
    //print hidden layers as vectors of weights and biases
    for(int i = 1; i < this->num_layers; i++){
        std::vector<Node> nodes = this->layers->at(i).get_nodes();
        for(int j = 0; j < this->layers->at(i).get_size(); j++){
            std::cout << "[";
            for(int k = 0; k < nodes[j].weights.size(); k++){
                std::cout << nodes[j].weights[k] << (k == nodes[i].weights.size() - 1 ? "": ", "); 
            }
            std::cout << "], " << nodes[j].bias << "\t";
        }
    }
}

//prints activations of network on given input
void Neural_Network::print_activated(std::vector<float> input){
    //first print inputs
    for(int i = 0; i < input.size(); i++){
        std::cout << input[i] << "\t";
    }
    std::cout << std::endl;
    //go through network, activate, and print activation
    std::vector<float> temp(input);
    for(int i = 1; i < this->layers->size(); i++){
        temp = this->layers->at(i).evaluate(temp);
        for(int j = 0; j < temp.size(); j++){
            std::cout << temp[j] << "\t";
        }
    }
}