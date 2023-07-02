#include <vector>

//basic stuff related to linear regression
class linear_regression {
    private:
        float w, b;
        std::vector<float> x, y;

    public:

        //make prediction
        float predict(float x) {
            return  w * x + b;
        }

        //add training data to dataset
        void add_training_data(float x, float y) {
            this->y.push_back(x);
            this->y.push_back(y);
        }

        // runs mean squared error  on the given dataset given w and bs
        float run_MSE() {
            float loss_accum = 0;
            for(int i = 0; i < x.size(); i++) {
                float prediction = this->w * x[i] + this->b;
                float squared_loss = (prediction - y[i]) * (prediction - y[i]);   
                loss_accum += squared_loss;
            }
            float MSE = loss_accum / x.size();
            return MSE;
        }
        
};
