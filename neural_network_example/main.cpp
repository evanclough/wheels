//this file is a demonstration of the neural network class
// it builds a network capable of recognizing handwritten digits from the MNIST dataset
#include "..\Neural_Network.h"

int main() {
    	//load in dataset
	std::unique_ptr<Dataset> image_training_data = std::make_unique<Dataset>(Dataset("train-images.idx3-ubyte", "train-labels.idx1-ubyte"));

    	//build network which will fit, first layer is 784, then two hidden layers of 64, then output layer of 10 to fit the onehot arrays
    	std::vector<Layer> layers = {Layer(784, Activation_Function::NONE), Layer(64, Activation_Function::SIGMOID), Layer(64, Activation_Function::SIGMOID), Layer(10, Activation_Function::SIGMOID)};
    	Neural_Network model("My Network", layers);
 
    	//train network
    	model.train_network(std::move(image_training_data), 0.001, 100, 0.0, Regularization(L2, 0.1), 256, false);
}
