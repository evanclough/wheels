//this file is a demonstration of the neural network class
// it builds a network capable of recognizing handwritten digits from the MNIST dataset
//currnetly this file is a benchmark of accuracy the different optimizers on the digits.
//it runs slow.. but its worth the wait i promise :)
#include "..\Neural_Network.h"

int main() {
    	//load in training dataset
	std::unique_ptr<Dataset> image_training_data = std::make_unique<Dataset>(Dataset("train-images.idx3-ubyte", "train-labels.idx1-ubyte", 100));
	std::unique_ptr<Dataset> image_test_data = std::make_unique<Dataset>(Dataset("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte", 100));
    	//build network which will fit, first layer is 784, then two hidden layers of 64, then output layer of 10 to fit the onehot arrays
    	std::vector<Layer> layers = {Layer(784, Activation_Function::NONE), Layer(64, Activation_Function::SIGMOID), Layer(10, Activation_Function::SIGMOID)};
		Neural_Network plain_model("Plain Model", layers);
		Neural_Network momentum_model("Momentum Model", layers);
		Neural_Network rms_prop_model("RMS Prop model", layers);
		Neural_Network adam_model("Adam Model", layers);
 
    	//train networks
    	plain_model.train_network(std::move(image_training_data), 0.001, 10, 0.0, new L2(0.1), 100, false, new No_Optimization());
		momentum_model.train_network(std::move(image_training_data), 0.001, 10, 0.0, new L2(0.1), 100, false, new Momentum(0.1));
    	rms_prop_model.train_network(std::move(image_training_data), 0.001, 10, 0.0, new L2(0.1), 100, false, new RMS_Prop(0.9));
		adam_model.train_network(std::move(image_training_data), 0.001, 10, 0.0, new L2(0.1), 100, false, new Adam());

	std::vector<Neural_Network*> models = {&plain_model, &momentum_model, &rms_prop_model, &adam_model};
	//for each model, test vs test data
	for(int i = 0; i < models.size(); i++){
		//test accuracy rate of network by running inferences on new image vectors and comparing output to label
		std::vector<std::vector<float>> test_feature_data = image_test_data->get_feature_data(), test_label_data = image_test_data->get_label_data(), inferences;
		for(int j = 0; j < test_feature_data.size(); j++){
			inferences.push_back(models[i]->inference(test_feature_data[i]));
		}

		float accuracy = 0;
		for(int j = 0; i < test_label_data.size(); j++){
			bool same = false;
			for(int k = 0; k < 10; k++){
				same |= ((int)inferences[j][k] & (int)test_label_data[j][k]);
			}
			accuracy += same;
		}
		accuracy /= test_label_data.size();
		std::cout << "Test Accuracy of " << models[i]->get_model_name() << ": " << accuracy * 100 << "%" << std::endl;
	}

	

}
