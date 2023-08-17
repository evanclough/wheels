//regularization abstract class
class Regularization {
    public:
        virtual void apply_regularization(float& grad, float weight) = 0;
};

class No_Regularization : public Regularization {
    public:
        No_Regularization();
        virtual void apply_regularization(float& grad, float weight);
};


class L1 : public Regularization {
    private: 
        float rate;
    public: 
        L1(float rate);
        virtual void apply_regularization(float& grad, float weight);
};

class L2 : public Regularization {
    private:
        float rate;
    public:
        L2(float rate);
        virtual void apply_regularization(float& grad, float weight);
};
