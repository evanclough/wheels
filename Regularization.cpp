//regularization file holds regularization classses
#include <stdexcept>
#include "Regularization.h"

//L1 and L2 regularization constructors takes in a rate, no reg constructor take snothign
No_Regularization::No_Regularization(){}
L1::L1(float rate){
    //check rate to see if valid
    if(rate < 0 || rate > 1){
        throw std::invalid_argument("To use regularization, you must pass in a  valid regularization rate.");
    }
    this->rate = rate;
}
L2::L2(float rate){
    //check rate to see if valid
    if(rate < 0 || rate > 1){
        throw std::invalid_argument("To use regularization, you must pass in a  valid regularization rate.");
    }
    this->rate = rate;
}

//L1 and L2 application methods apply respective regularization equations to weight gradient, no reg is blank
void No_Regularization::apply_regularization(float& grad, float weight){}
void L1::apply_regularization(float& grad, float weight){
    grad += this->rate * (weight > 0) ? 1 : -1;
}

void L2::apply_regularization(float& grad, float weight){
    grad += this->rate * weight;
}