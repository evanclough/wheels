#include <vector>
#include <memory>

class Linear_Regression {
    private:
        float w, b;
        std::unique_ptr<std::vector<float>> x_data, y_data;

    public:
    
        float predict(float x);
        void add_training_data(float x, float y);
        float run_MSE();
        Linear_Regression();
        Linear_Regression(std::vector<float> initial_x_data, std::vector<float> initial_y_data);
        Linear_Regression(std::vector<float> initial_x_data, std::vector<float> initial_y_data, float w, float b);
};