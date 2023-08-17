//optimizers file stores main parent class for optimizers and individual optimizer classes
#include <vector>

class Optimizer {
    public:
        //generic methods to update the weights/biases, takes in weight/bias gradient to adjust, and persistent weight/bias matrices to uose in calculation/update 
        virtual void initialize_persistent_values(std::vector<std::vector<std::vector<std::vector<float>>>>& persistent_values, int num_layers, std::vector<int> layer_sizes, std::vector<std::vector<int>> weights_sizes) = 0;
        virtual void grad_update(std::vector<std::vector<std::vector<float>>>& weights_grad, std::vector<std::vector<float>>& biases_grad, std::vector<std::vector<std::vector<std::vector<float>>>> &persistent_values) = 0;
};

//default no optimizer class with blank update methods
class No_Optimization : public Optimizer {
    public:
        No_Optimization();

        //update methods 
        virtual void initialize_persistent_values(std::vector<std::vector<std::vector<std::vector<float>>>>& persistent_values, int num_layers, std::vector<int> layer_sizes, std::vector<std::vector<int>> weights_sizes);
        virtual void grad_update(std::vector<std::vector<std::vector<float>>>& weights_grad, std::vector<std::vector<float>>& biases_grad, std::vector<std::vector<std::vector<std::vector<float>>>> &persistent_values);
};

//Momentum optimizer utliizes the previous gradient values in adjusting the learning rate
class Momentum : public Optimizer {
    private:
        float rate;
    public: 
        //constructor forces user to set momentum rate
        Momentum(float rate);

        //update methods 
        virtual void initialize_persistent_values(std::vector<std::vector<std::vector<std::vector<float>>>>& persistent_values, int num_layers, std::vector<int> layer_sizes, std::vector<std::vector<int>> weights_sizes);
        virtual void grad_update(std::vector<std::vector<std::vector<float>>>& weights_grad, std::vector<std::vector<float>>& biases_grad, std::vector<std::vector<std::vector<std::vector<float>>>> &persistent_values);

};

//RMS_Prop optimizer adjusts learning rate based on mean squared average of the gradients
class RMS_Prop : public Optimizer {
    private:
        float beta = 0.9;
    public:
        //blank constructor leaves default hyperparameter
        RMS_Prop();
        //constructor with param allows user to set beta
        RMS_Prop(float beta);

        //update methods
        virtual void initialize_persistent_values(std::vector<std::vector<std::vector<std::vector<float>>>>& persistent_values, int num_layers, std::vector<int> layer_sizes, std::vector<std::vector<int>> weights_sizes);
        virtual void grad_update(std::vector<std::vector<std::vector<float>>>& weights_grad, std::vector<std::vector<float>>& biases_grad, std::vector<std::vector<std::vector<std::vector<float>>>> &persistent_values);
};

//adam optimizer combines aspects of rms_prop and momentum to adjust learning rate
class Adam : public Optimizer{
    private:
        float beta1 = 0.9;
        float beta2 = 0.999;
        float epsilon = 1e-8;
    public:

        //blank constructor leaves default hyperparameters
        Adam();

        //constructor with params allows user to set hyperparams
        Adam(float beta1, float beta2, float epsilon);

        //update methods
        virtual void initialize_persistent_values(std::vector<std::vector<std::vector<std::vector<float>>>>& persistent_values, int num_layers, std::vector<int> layer_sizes, std::vector<std::vector<int>> weights_sizes);
        virtual void grad_update(std::vector<std::vector<std::vector<float>>>& weights_grad, std::vector<std::vector<float>>& biases_grad, std::vector<std::vector<std::vector<std::vector<float>>>> &persistent_values);
};