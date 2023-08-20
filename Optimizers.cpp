//Optimizer file initializes constructors for the different optimizer structs

#include "Optimizers.h"
#include <stdexcept>
#include <cmath>
#include <iostream>

//no optimizatio constructor does nothing
No_Optimization::No_Optimization(){}
//no optimization update methods do nothing
void No_Optimization::initialize_persistent_values(std::vector<std::vector<std::vector<std::vector<float>>>>& persistent_values, int num_layers, std::vector<int> layer_sizes, std::vector<std::vector<int>> weights_sizes){}
void No_Optimization::grad_update(std::vector<std::vector<std::vector<float>>>& weights_grad, std::vector<std::vector<float>>& biases_grad, std::vector<std::vector<std::vector<std::vector<float>>>> &persistent_values){}

//basic momentum constructor takes in rate parameters, checks it, and sets it
Momentum::Momentum(float rate){
    //check if rate in range
    if(rate < 0 || rate > 1){
        throw std::invalid_argument("passed momentum rate must be between zero and one.");
    }
    this->rate = rate;
}

//momentum update methods 
void Momentum::initialize_persistent_values(std::vector<std::vector<std::vector<std::vector<float>>>>& persistent_values, int num_layers, std::vector<int> layer_sizes, std::vector<std::vector<int>> weights_sizes){
    //check if already initialized, if not, initialize according to passed dimensional params
    if(!persistent_values.size()){
        std::vector<std::vector<std::vector<float>>> prev_weights_grad;
        std::vector<std::vector<float>> prev_biases_grad;
        for(int i = 0 ; i < num_layers; i++){
            prev_weights_grad.push_back({});
            prev_biases_grad.push_back({});
            for(int j = 0; j < layer_sizes[i]; j++){
                prev_weights_grad[i].push_back({});
                prev_biases_grad[i].push_back(0);
                for(int k = 0; k < weights_sizes[i][j]; k++){
                    prev_weights_grad[i][j].push_back(0);
                }
            }
        }
        persistent_values.push_back(prev_weights_grad);
        persistent_values.push_back({prev_biases_grad});
    }
}
void Momentum::grad_update(std::vector<std::vector<std::vector<float>>>& weights_grad, std::vector<std::vector<float>>& biases_grad, std::vector<std::vector<std::vector<std::vector<float>>>> &persistent_values){
    //first pull relevant info from persistent data, which for momentum is the previous weight and bias gradients
    std::vector<std::vector<std::vector<float>>>* prev_weights_grad = &persistent_values[0];
    std::vector<std::vector<float>>* prev_biases_grad = &persistent_values[1][0];

    //update gradient according to specified rate of momentum times the previous gradient value, then update 
    //the persistent value to be used in the next calculation
    for(int i = 0; i < weights_grad.size(); i++){
		for(int j = 0; j < weights_grad[i].size(); j++){
			biases_grad[i][j] += this->rate * (*prev_biases_grad)[i][j];
            (*prev_biases_grad)[i][j] = biases_grad[i][j];
			for(int k = 0; k < weights_grad[i][j].size(); k++){
    			weights_grad[i][j][k] += this->rate * (*prev_weights_grad)[i][j][k];
                (*prev_weights_grad)[i][j][k] = weights_grad[i][j][k];
	    	}
		}
   	}
}

//basic RMS prop constructor leaves beta as the default
RMS_Prop::RMS_Prop(){}

//RMS prop constructo wtih param checks and sets beta
RMS_Prop::RMS_Prop(float beta){
    //check if in range
    if(beta < 0 || beta > 1){
        throw std::invalid_argument("passed beta value must be ebtween zero and one.");
    }
    this->beta = beta;
}

//RMS Prop update methods
void RMS_Prop::initialize_persistent_values(std::vector<std::vector<std::vector<std::vector<float>>>>& persistent_values, int num_layers, std::vector<int> layer_sizes, std::vector<std::vector<int>> weights_sizes){
    //check if already initialized, if not, initialize according to passed dimensional params
    if(!persistent_values.size()){
        std::vector<std::vector<std::vector<float>>> weights_RMS_average;
        std::vector<std::vector<float>> biases_RMS_average;
        for(int i = 0 ; i < num_layers; i++){
            weights_RMS_average.push_back({});
            biases_RMS_average.push_back({});
            for(int j = 0; j < layer_sizes[i]; j++){
                weights_RMS_average[i].push_back({});
                biases_RMS_average[i].push_back(0);
                for(int k = 0; k < weights_sizes[i][j]; k++){
                    weights_RMS_average[i][j].push_back(0);
                }
            }
        }
        persistent_values.push_back(weights_RMS_average);
        persistent_values.push_back({biases_RMS_average});
    }
}

void RMS_Prop::grad_update(std::vector<std::vector<std::vector<float>>>& weights_grad, std::vector<std::vector<float>>& biases_grad, std::vector<std::vector<std::vector<std::vector<float>>>> &persistent_values){
    //pull info from persistent data, which for RMS prop is the moving average of gradients
    std::vector<std::vector<std::vector<float>>>* weights_grad_avg = &persistent_values[0];
    std::vector<std::vector<float>>* biases_grad_avg = &persistent_values[1][0];

    //update gradient according to RMS_Prop formula, divide it by the square root of the squared moving average of the gradients
    // then also update hte moving average of the gradients with the newly calculated one
    for(int i = 0; i < weights_grad.size(); i++){
		for(int j = 0; j < weights_grad[i].size(); j++){
			biases_grad[i][j] /= (std::sqrt((this->beta * (*biases_grad_avg)[i][j]) + ((1 - this->beta) * (biases_grad[i][j] * biases_grad[i][j]))) == 0 ? 1e-8 : std::sqrt((this->beta * (*biases_grad_avg)[i][j]) + ((1 - this->beta) * (biases_grad[i][j] * biases_grad[i][j]))));			
            (*biases_grad_avg)[i][j] += (1 - this->beta) * (biases_grad[i][j] * biases_grad[i][j]);
			for(int k = 0; k < weights_grad[i][j].size(); k++){
				weights_grad[i][j][k] /= (std::sqrt((this->beta * (*weights_grad_avg)[i][j][k]) + ((1 - this->beta) * (weights_grad[i][j][k] * weights_grad[i][j][k]))) == 0 ? 1e-8 : std::sqrt((this->beta * (*weights_grad_avg)[i][j][k]) + ((1 - this->beta) * (weights_grad[i][j][k] * weights_grad[i][j][k]))));
                (*weights_grad_avg)[i][j][k] += (1 - this->beta) * (weights_grad[i][j][k] * weights_grad[i][j][k]);
	    	}
		}
   	}
}

//basic Adam constructor leaves params as default
Adam::Adam(){}

//Adam constructor with params checks and sets the,
Adam::Adam(float beta1, float beta2, float epsilon){
    //check if in range
    if(beta1 < 0 || beta1 > 1){
        throw std::invalid_argument("passed beta1 value must be between zero and one.");
    }

    //check if in range
    if(beta2 < 0 || beta2 > 1){
        throw std::invalid_argument("passed beta2 value must be between zero and one.");
    }

    //check if in range
    if(epsilon < 0 || epsilon > 1){
        throw std::invalid_argument("passed epsilon value must be between zero and one.");
    }

    this->beta1 = beta1;
    this->beta2 = beta2;
    this->epsilon = epsilon;
}

//Adam update methods
void Adam::initialize_persistent_values(std::vector<std::vector<std::vector<std::vector<float>>>>& persistent_values, int num_layers, std::vector<int> layer_sizes, std::vector<std::vector<int>> weights_sizes){
    //check if already initialized, if not, initialize according to passed dimensional params
    if(!persistent_values.size()){
        std::vector<std::vector<std::vector<float>>> vt_weights, st_weights;
        std::vector<std::vector<float>> vt_biases, st_biases;
        for(int i = 0 ; i < num_layers; i++){
            vt_weights.push_back({});
            st_weights.push_back({});
            vt_biases.push_back({});
            st_biases.push_back({});
            for(int j = 0; j < layer_sizes[i]; j++){
                vt_weights[i].push_back({});
                st_weights[i].push_back({});
                vt_biases[i].push_back(0);
                st_biases[i].push_back(0);
                for(int k = 0; k < weights_sizes[i][j]; k++){
                    vt_weights[i][j].push_back(0);
                    st_weights[i][j].push_back(0);
                }
            }
        }
        persistent_values.push_back(vt_weights);
        persistent_values.push_back(st_weights);
        persistent_values.push_back({vt_biases});
        persistent_values.push_back({st_biases});
    }
}

void Adam::grad_update(std::vector<std::vector<std::vector<float>>>& weights_grad, std::vector<std::vector<float>>& biases_grad, std::vector<std::vector<std::vector<std::vector<float>>>> &persistent_values){
    //pull info from persistent data, which for RMS prop is the moving average of gradients
    std::vector<std::vector<std::vector<float>>>* vt_weights = &persistent_values[0];
    std::vector<std::vector<std::vector<float>>>* st_weights = &persistent_values[1];
    std::vector<std::vector<float>>* vt_biases = &persistent_values[2][0];
    std::vector<std::vector<float>>* st_biases = &persistent_values[3][0];

    //update gradient according to Adam optimization formula, the average of the moments of the gradients
    //also update persistent values to be used in future calculation
    for(int i = 0; i < weights_grad.size(); i++){
		for(int j = 0; j < weights_grad[i].size(); j++){
            (*vt_biases)[i][j] = (this->beta1 * (*vt_biases)[i][j]) + (1 - this->beta1) * biases_grad[i][j];
			(*st_biases)[i][j] = (this->beta2 * (*st_biases)[i][j]) + (1 - this->beta2) * (biases_grad[i][j] * biases_grad[i][j]);
			biases_grad[i][j] *= ((*vt_biases)[i][j] / (1 - this->beta1));
			biases_grad[i][j] /= std::sqrt(((*st_biases)[i][j] / (1 - this->beta2)) + this->epsilon);
			for(int k = 0; k < weights_grad[i][j].size(); k++){
				(*vt_weights)[i][j][k] = (this->beta1 * (*vt_weights)[i][j][k]) + (1 - this->beta1) * weights_grad[i][j][k];
				(*st_weights)[i][j][k] = (this->beta2 * (*st_weights)[i][j][k]) + (1 - this->beta2) * (weights_grad[i][j][k] * weights_grad[i][j][k]);
				weights_grad[i][j][k] *= ((*vt_weights)[i][j][k] / (1 - this->beta1));
				weights_grad[i][j][k] /= std::sqrt(((*st_weights)[i][j][k] / (1 - this->beta2)) + this->epsilon);
	    	}
		}
   	}
}