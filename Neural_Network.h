//hthe neural network class allows for creation and training of a simple neural network

#include <vector>
#include <string>
#include <memory>
#include "Layer.h"
#include "Dataset.h"

//regularization type enum allows user to specify the regularization type that they want
enum Regularization_Type {
	L1,
	L2
};

//regularization struct must be passed in to train_network function to specify the regularization used in training model 
struct Regularization {
	Regularization_Type reg_type;
	float rate;
	Regularization(Regularization_Type reg_type, float rate);
};

class Neural_Network {
    private:
        int num_layers, num_features;
        std::string model_name;
        std::unique_ptr<std::vector<Layer>> layers;

        std::vector<std::vector<float>> temp_activations;
        std::vector<std::vector<float>> temp_z_values;

        //calculates the patial derivative of the cost function of the network with respect to a given weight 
        float pd_weight(std::vector<float> feature, std::vector<float> label, int layer, int j, int k);

        //partial derivative of z value with respect to given weihgt, recursive helpyter to pd_weight
        float pd_z_wrt_weight(int weight_layer, int weight_j, int weight_k, int z_layer, int z_j);

        //calculates the patial derivative of the cost function of the network with respect to a given bias 
        float pd_bias(std::vector<float> feature, std::vector<float> label, int layer, int j);

        //finds partial derivative of a given z value with respect to a given weight, recursive helpyer to pd_weight
        float pd_z_wrt_bias(int bias_layer, int bias_j, int z_layer, int z_j);
    public:
        //default constructor
        Neural_Network(std::string model_name, std::vector<Layer> layers);

        //sets given weight
        void set_weight(int layer, int j, int k, float weight);

        // sets given bias
        void set_bias(int layer, int j, float bias);

        //runs neural network with a set of input data and returns output
        std::vector<float> inference(std::vector<float> input);

        //fetches activations of network on given input
        std::vector<std::vector<float>> activations(std::vector<float> input);

        //fetches unactivated layers
        std::vector<std::vector<float>> z_values(std::vector<float> input);

        //runs MSE loss on given dataset
        float run_MSE(std::vector<std::vector<float>> feature_data, std::vector<std::vector<float>> label_data);

        //runs backpropogation on network given feature and label vectors
        void gradient_descent(std::vector<std::vector<float>> features, std::vector<std::vector<float>> labels, float learning_rate, Regularization regularization);
        
        //derivative of a given activation function
        float activation_derivative(float input, Activation_Function activation);

        //trains network with a given training dataset, learning rate, number of epochs, and validation split
        void train_network(std::unique_ptr<Dataset> training_data, float learning_rate, int epochs, float validation_split, Regularization regularization, int batch_size, bool print_weights);

        //tests network on given test data set and returns error
        void test_network(std::unique_ptr<Dataset> test_data);

        //print network with weights and biases in terminal
        void print_network();

        //print activations of network on given input
        void print_activated(std::vector<float> input);
};
