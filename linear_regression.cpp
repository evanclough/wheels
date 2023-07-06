#include <vector>
#include <memory>
#include "Linear_Regression.h"

        //basic stuff related to linear regression
        float w, b;
        std::unique_ptr<std::vector<float>> x_data, y_data;


        //constructors
        Linear_Regression::Linear_Regression() {
            this->w = 0;
            this->b = 0;
            this->x_data = std::make_unique<std::vector<float>>();
            this->y_data = std::make_unique<std::vector<float>>();
        }

        Linear_Regression::Linear_Regression(std::vector<float> initial_x_data, std::vector<float> initial_y_data) {
            this->w = 0;
            this->b = 0;
            this->x_data = std::make_unique<std::vector<float>>(initial_x_data);
            this->y_data = std::make_unique<std::vector<float>>(initial_y_data);
        }

        Linear_Regression::Linear_Regression(std::vector<float> initial_x_data, std::vector<float> initial_y_data, float w, float b){
            this->w = w;
            this->b = b;
            this->x_data = std::make_unique<std::vector<float>>(initial_x_data);
            this->y_data = std::make_unique<std::vector<float>>(initial_y_data);
        }

        //make prediction
        float Linear_Regression::predict(float x) {
            return  w * x + b;
        }

        //add training data to dataset
        void Linear_Regression::add_training_data(float x, float y) {
            this->y_data->push_back(x);
            this->y_data->push_back(y);
        }

        // runs mean squared error  on the given dataset given w and bs
        float Linear_Regression::run_MSE() {
            float loss_accum = 0;
            for(int i = 0; i < this->x_data->size(); i++) {
                float prediction = this->w * (*this->x_data)[i] + this->b;
                float squared_loss = (prediction - (*this->y_data)[i]) * (prediction - (*this->y_data)[i]);   
                loss_accum += squared_loss;
            }
            float MSE = loss_accum / this->x_data->size();
            return MSE;
        }