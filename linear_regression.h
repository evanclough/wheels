#include <vector>

class linear_regression {
    private:
        float w, b;
        std::vector<float> x, y;

    public:
        float predict();
        void add_training_data(float x, float y);
        void run_MSE();
};