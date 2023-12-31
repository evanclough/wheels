//hthe neural network class allows for the creation of layers to be used by the neural network
#include <vector>
#include <memory>
#include <iostream>

enum class Activation_Function {
    RELU, 
    TANH,
    SIGMOID,
    NONE
};

struct Node {
    std::vector<float> weights;
    float bias;
};

class Layer {
    private:
        int size;
        Activation_Function activation;
        //a list of nodes, each with a list of weights, and a bias, to be trated as a tuple
        std::unique_ptr<std::vector<Node>> nodes; 
    public:
        //basic constructor takes in layer size
        Layer(int size, Activation_Function activation);
        Layer(const Layer& cpy);
        //getters and setters
        int get_size();
        std::vector<Node> get_nodes();
        Activation_Function get_activation();

        //sets weight arrays of all nodes to a given size and constant val
        void set_weights(int size, float weight);

        //sets given weight to given value
        void set_weight(int i, int j, float weight);

        //sets bieases ofa ll nodes to a given val
        void set_biases(float bias);

        //sets given bias to given value
        void set_bias(int index, float bias);

        //sets nodees array to all have a given number of weights initialized to 0, and sets all biases to 0
        void set_default(int size);

        //evalutates layer with given input and an activation function
        std::vector<float> evaluate(std::vector<float> input);

        //evaluates layer with no activation
        std::vector<float> evaluate_without_activation(std::vector<float> input);
}; 

