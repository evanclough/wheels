//this file is a demonstration of the neural network class
// it builds a network capable of recognizing handwritten digits from the MNIST dataset
#include "..\Neural_Network.h"

int main() {
    	//load in training dataset
	std::unique_ptr<Dataset> image_training_data = std::make_unique<Dataset>(Dataset("train-images.idx3-ubyte", "train-labels.idx1-ubyte"));
	std::unique_ptr<Dataset> image_test_data = std::make_unique<Dataset>(Dataset("test-images.idx3-ubyte", "test-labels.idx1-ubyte"));
    	//build network which will fit, first layer is 784, then two hidden layers of 64, then output layer of 10 to fit the onehot arrays
    	std::vector<Layer> layers = {Layer(784, Activation_Function::NONE), Layer(64, Activation_Function::SIGMOID), Layer(64, Activation_Function::SIGMOID), Layer(10, Activation_Function::SIGMOID)};
    	Neural_Network model("My Network", layers);
 
    	//train network
    	model.train_network(std::move(image_training_data), 0.001, 100, 0.0, Regularization(Regularization_Type::L2, 0.2), 256, false, Optimizer(Optimizer_Type::MOMENTUM, 0.2));

	//test accuracy rate of network by running inferences on new image vectors and comparing output to label
	std::vector<std::vector<float>> test_feature_data = image_test_data->get_feature_data(), test_label_data = image_test_data->get_label_data(), inferences;
	for(int i = 0; i < test_feature_data.size(); i++){
		inferences.push_back(model.inference(test_feature_data[i]));
	}

	float accuracy = 0;
	for(int i = 0; i < test_label_data.size(); i++){
		bool same = false;
		for(int j = 0; j < 10; j++){
			same |= ((int)inferences[i][j] & (int)test_label_data[i][j]);
		}
		accuracy += same;
	}
	accuracy /= test_label_data.size();
	std::cout << "Test Accuracy: " << accuracy * 100 << "%" << std::endl;

}
