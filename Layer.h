//hthe neural network class allows for the creation of layers to be used by the neural network
#include <vector>

struct Node {
    std::vector<float> weights;
    float bias;
};

class Layer {
    private:
        //a list of nodes, each with a list of weights, and a bias, to be trated as a tuple
        std::vector<Node> nodes; 
    public:
        void activate(Activation_Function);
}; 

enum Activation_Function {
    RELU, 
    TANH,
    SIGMOID,
    NONE
};