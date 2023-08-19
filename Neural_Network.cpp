#include "Neural_Network.h"
#include <stdexcept>
#include <math.h>
#include <limits>

//basic constructor takes a model name and a layer array to create network
Neural_Network::Neural_Network(std::string model_name, std::vector<Layer> layers){

    this->layers = std::make_unique<std::vector<Layer>>(layers);
    //check if layers array has at least two layers, throw error otherwise.
    if(layers.size() < 2){
        throw std::invalid_argument("Layers array must include at least two layers.");
    }
    this->num_layers = this->layers->size();
    this->num_features = this->layers->at(0).get_size();

    //initialize weights and biases in layers to 0
    for(int i = 1; i < this->num_layers; i++){
        this->layers->at(i).set_default(this->layers->at(i - 1).get_size());
    }
    this->model_name = model_name == "" ? "Model" : model_name;
}
//sets given weight
void Neural_Network::set_weight(int layer, int j, int k, float weight){
    //quick check of passed in layer
    if(layer < 1 || layer >= this->num_layers){
        throw std::invalid_argument("please enter valid layer to set the weight of,");
    }
    this->layers->at(layer).set_weight(j, k, weight);
}

// sets given bias
void Neural_Network::set_bias(int layer, int j, float bias){
    //quick check of passed in layer
    if(layer < 1 || layer >= this->num_layers){
        throw std::invalid_argument("please enter valid layer to set the bias of,");
    }
    this->layers->at(layer).set_bias(j, bias);
}

//runs neural network with given input data
std::vector<float> Neural_Network::inference(std::vector<float> input){
    std::vector<float> temp = input;
    for(int i = 1; i < this->num_layers; i++){
        temp = this->layers->at(i).evaluate(temp);
    }
    return temp;
}

//gets model name
std::string Neural_Network::get_model_name(){
    return this->model_name;
}

//fetches matrix of activations of network on given input
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

//fetches unactivated layers to help with the training function
std::vector<std::vector<float>> Neural_Network::z_values(std::vector<float> input){
    std::vector<std::vector<float>> activations = this->activations(input);
    std::vector<std::vector<float>> accum = {input};
    for(int i = 1; i < activations.size(); i++){
        accum.push_back(this->layers->at(i).evaluate_without_activation(activations[i - 1]));
    }
    return accum;
}

//runs MSE loss on given dataset
float Neural_Network::run_MSE(std::vector<std::vector<float>> feature_data, std::vector<std::vector<float>> label_data){
    std::vector<std::vector<float>> predicted;
    for(int i = 0; i < feature_data.size(); i++){
        predicted.push_back(this->inference(feature_data[i]));
    }
    std::vector<float> loss_accum(label_data[0].size(), 0);
    for(int i = 0; i < feature_data.size(); i++){
        for(int j = 0; j < label_data[0].size(); j++){
            loss_accum[j] += (predicted[i][j] - label_data[i][j]) * (predicted[i][j] - label_data[i][j]);
        }
    }
    for(int i = 0; i < label_data[0].size(); i++){
        loss_accum[i] /= feature_data.size();
    }

    float final_loss_average = 0;
    for(int i = 0; i < loss_accum.size(); i++){
        final_loss_average += (loss_accum[i] / loss_accum.size());
    }
    return final_loss_average;
}

//gradient descent runs backpropagation on given feature and label set with given learning rate, regularization, and optimizer
void Neural_Network::gradient_descent(std::vector<std::vector<float>> features, std::vector<std::vector<float>> labels, float learning_rate, Regularization* regularization, Optimizer* optimizer, std::vector<std::vector<std::vector<std::vector<float>>>> &persistent_values){
    //initialize weight and bias gradient matrices
    std::vector<std::vector<std::vector<float>>> weights_grad;
    std::vector<std::vector<float>> biases_grad;

    //initialize gradient matrices to zero
    for(int i = 0; i < this->num_layers; i++){
        weights_grad.push_back({});
        biases_grad.push_back({});
        for(int j = 0; j < this->layers->at(i).get_size(); j++){
            weights_grad[i].push_back({});
            biases_grad[i].push_back(0);
            for(int k = 0; k < this->layers->at(i).get_nodes()[j].weights.size(); k++){
                weights_grad[i][j].push_back(0);
            }
        }
    }

    //for each feature label pair in the batch, calculate the partial derivative of the cost function with 
    //with respect to each weight and bias in the network, and keep track of the average
    //of all of them in this batch
    for(int data_pair = 0; data_pair < labels.size(); data_pair++){
        //fetch activations and z values for whole network for this example
        this->temp_activations = this->activations(features[data_pair]);
        this->temp_z_values = this->z_values(features[data_pair]);
        this->temp_activation_derivs = temp_z_values;
        for(int i = 0; i < temp_activation_derivs.size(); i++){
            for(int j = 0; j < temp_activation_derivs[i].size(); j++){
                this->temp_activation_derivs[i][j] = this->activation_derivative(this->temp_activation_derivs[i][j], this->layers->at(i).get_activation());
            }
        }
        std::vector<std::vector<std::vector<std::vector<float>>>> grads = this->grads(features[data_pair], labels[data_pair]);

        for(int i = this->layers->size() - 1; i > 0; i--){
            for(int j = 0; j < this->layers->at(i).get_size(); j++){
                biases_grad[i][j] += (1.0 / labels.size()) * grads[1][0][i - 1][j];
                for(int k = 0; k < this->layers->at(i).get_nodes()[j].weights.size(); k++){
                    weights_grad[i][j][k] += (1.0 / labels.size()) * grads[0][i - 1][j][k]; 
		        }
            }
        }

        //reset temp activations and z values array
        this->temp_activations = {};
        this->temp_z_values = {};
        this->temp_activation_derivs = {};
    }

    //adjust weights gradient matrix according to specified regularization
            for(int i = this->layers->size() - 1; i > 0; i--){
                for(int j = 0; j < this->layers->at(i).get_size(); j++){
                    for(int k = 0; k < this->layers->at(i).get_nodes()[j].weights.size(); k++){
                        regularization->apply_regularization(weights_grad[i][j][k], this->layers->at(i).get_nodes()[j].weights[k]);
                    }
                }
            }

    //adjust weights and biases matrices according to specified optimization method
    optimizer->grad_update(weights_grad, biases_grad, persistent_values);

    //adjust previous weights according to average of gradient of training examples
    for(int i = this->layers->size() - 1; i > 0; i--){
        for(int j = 0; j < this->layers->at(i).get_size(); j++){
            this->layers->at(i).set_bias(j, this->layers->at(i).get_nodes()[j].bias - learning_rate * biases_grad[i][j]);
            for(int k = 0; k < this->layers->at(i).get_nodes()[j].weights.size(); k++){
                this->layers->at(i).set_weight(j, k, this->layers->at(i).get_nodes()[j].weights[k] - learning_rate * weights_grad[i][j][k]);
            }
        }
    }
}

//derivative of the current activation function of the network
float Neural_Network::activation_derivative(float input, Activation_Function activation){
    switch(activation){
	case Activation_Function::SIGMOID:
            return ((1 / (1 + std::pow(2.71828, -input)))) * (1 - ((1 / (1 + std::pow(2.71828, -input)))));
        break;
	case Activation_Function::TANH:
            return 1 - (std::tanh(input) * std::tanh(input));
        break;
	case Activation_Function::RELU:
            return input < 0 ? 0 : 1;
        break;
        default:
            return input;
    }
}

std::vector<std::vector<std::vector<std::vector<float>>>> Neural_Network::grads(std::vector<float> feature, std::vector<float> label){    
    //initialize and fill in gradient wtih zeroes to start
    std::vector<std::vector<std::vector<float>>> weights_grad;
    std::vector<std::vector<float>> biases_grad;
    for(int i = 0; i < this->num_layers - 1; i++){
        weights_grad.push_back({});
        biases_grad.push_back({});
        for(int j = 0; j < this->layers->at(i + 1).get_size(); j++){
            weights_grad[i].push_back({});
            biases_grad[i].push_back(0);
            for(int k = 0; k < this->layers->at(i).get_size(); k++){
                weights_grad[i][j].push_back(0);
            }
        }
    }

   //final layer to be multiplied by accumulation of previous layers caluclated for each
    std::vector<float> final_layer;
    for(int i = 0; i < label.size(); i++){
        final_layer.push_back(2 * (this->temp_activations[this->num_layers - 1][i] - label[i]) * this->temp_activation_derivs[this->num_layers - 1][i]);
    }

    for(int i = this->num_layers - 1; i >= 1; i--){
        //current layer to be used in calculation, initialize it as the initial layer accum for when 
        //none needs to be done
            std::vector<float> prev_arr(this->layers->at(i).get_size(), 1);
            for(int j = i; j < this->num_layers - 1; j++){
                std::vector<float> new_arr(this->layers->at(j + 1).get_size(), 0);
                for(int k = 0; k < new_arr.size(); k++){
                    for(int l = 0; l < prev_arr.size(); l++){
                        new_arr[k] +=  this->temp_activation_derivs[j + 1][k] * this->layers->at(j + 1).get_nodes()[k].weights[l] * prev_arr[l];//weight * activation derivative * prev_arr value
                    }
                }
                prev_arr = new_arr;
            }
        float sum = 0;
        for(int j = 0; j < prev_arr.size(); j++){
            sum += prev_arr[j] * final_layer[j];

        }

        sum /= label.size();
        
        //set biases
        for(int j = 0; j < this->layers->at(i).get_size(); j++){
            biases_grad[i - 1][j] = sum;
        }

        //set weights for this layer to activation of attached node times the sum we've accumulated
        for(int j = 0; j < this->layers->at(i - 1).get_size(); j++){
            for(int k = 0; k < this->layers->at(i).get_size(); k++){
                weights_grad[i - 1][k][j] = sum * this->temp_activations[i - 1][j];
            }
        }
    }
    return {weights_grad, {biases_grad}};
};

//trains network with a given training dataset, learning rate, number of epochs, validation split, batch size, and optimizer
//pretty mucht he same as the linear regression training function with some small changes
void Neural_Network::train_network(std::unique_ptr<Dataset> training_data, float learning_rate, int epochs, float validation_split, Regularization* regularization, int batch_size, bool print_state, Optimizer* optimizer){
    
    //check to see if batch size at least one
    if(batch_size < 1){
	throw std::invalid_argument("paramater batch_size must be at least 1.");
    }

    //check dataset dimensions to see if compatible with network
    if(training_data->get_feature_data()[0].size() != this->layers->at(0).get_size()){
        throw std::invalid_argument("The feature data size of the passed dataset does not match the size of the input layer of the network being trained.");
    }

    if(training_data->get_label_data()[0].size() != this->layers->at(this->num_layers - 1).get_size()){
        throw std::invalid_argument("The label data size of the passed dataset does not match the size of the output layer of the network being trained.");
    }

    //check if optimizer exists, if not, set to no reg
    if(regularization == nullptr){
        regularization = new No_Regularization();
    }

    //check if optimizer exists, if not, set to no optimization
    if(optimizer == nullptr){
        optimizer = new No_Optimization();
    }

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

        //print current state of network if set to
	if(print_state){
		this->print_network();
	}

    //run gradient descent with each feature/value pair in training dataset
    std::vector<std::vector<float>> feature_data = training_data->get_feature_data();
    std::vector<std::vector<float>> label_data = training_data->get_label_data();
	
	
	//run gradient descent for each batch
	for(int j = 0; j < feature_data.size() / batch_size + (feature_data.size() % batch_size != 0); j++){
        //iterators to specify batch to train on
        auto start_feature_it = feature_data.begin() + j * batch_size;
        auto end_feature_it = ((j + 1) * batch_size  < label_data.size() ? feature_data.begin() + (j + 1) * batch_size : feature_data.end());
        auto start_label_it = label_data.begin() + j * batch_size;
        auto end_label_it = ((j + 1) * batch_size  < label_data.size() ? label_data.begin() + (j + 1) * batch_size : label_data.end());

        //matrix of persistent values to be used in optimization, initialize with the optimizers respective intiialization method
        std::vector<std::vector<std::vector<std::vector<float>>>> persistent_values;
        //gather relevant dimensional values pass to method
        std::vector<int> layer_sizes;
        std::vector<std::vector<int>> weights_sizes;
        for(int i = 0; i < this->num_layers; i++){
            layer_sizes.push_back(this->layers->at(i).get_size());
            weights_sizes.push_back({});
            for(int j = 0; j < this->layers->at(i).get_nodes().size(); j++){
                weights_sizes[i].push_back(this->layers->at(i).get_nodes()[j].weights.size());
            }
        }
        optimizer->initialize_persistent_values(persistent_values, this->num_layers, layer_sizes, weights_sizes);
        this->gradient_descent(std::vector<std::vector<float>>(start_feature_it, end_feature_it), std::vector<std::vector<float>>(start_label_it, end_label_it), learning_rate, regularization, optimizer, persistent_values);
	}

        //print loss
        std::cout << "Training Loss: ";
        float training_loss = this->run_MSE(feature_data, label_data);
        std::cout << training_loss;
        std::cout << std::endl;

        //same for validation if there is any
        std::cout << "Validation Loss:";
        if(val_set_size == 0){
            std::cout << " N/A" << std::endl;
        }else {
            float validation_loss = this->run_MSE(validation_data->get_feature_data(), validation_data->get_label_data());
            std::cout << " " << validation_loss << " " << std::endl;
        }
    }
}

//tests network on given test data set and returns error
void Neural_Network::test_network(std::unique_ptr<Dataset> test_data){
    std::cout << "Testing " << this->model_name << "..." << std::endl;
    float test_loss = this->run_MSE(test_data->get_feature_data(), test_data->get_label_data());
    std::cout << "Test Loss: " << test_loss << std::endl;
}

//prints current weights and biases of network
void Neural_Network::print_network(){
    //print input layer
    for(int i = 0; i < this->layers->at(0).get_size(); i++){
        std::cout << " 0 " << "\t\t";
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
            std::cout << "], " << nodes[j].bias << "\t\t";
        }
        std::cout << std::endl;
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
