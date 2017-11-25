#ifndef DNN_HPP
#define DNN_HPP
#include <stdio.h>
#include "global_def.hpp"

using namespace global_def;
class DNN {
public:
    DNN(size_t i_n_layers, size_t* i_stuc_layers, size_t n_params, double* params
        , double std_dev = 1.0, int regularizer = global_def::L2);
    ~DNN();
    void initialize(double std_dev);
    void print_all();
    // passing increments
    void update_parameters(std::vector<Tuple> tuples);

    double zero_oracle(Batch batch);
    std::vector<Tuple> first_oracle(Batch batch);
    std::vector<Tuple> hessian_vector_oracle(Batch batch, MatrixXr* V);
private:
    size_t n_layers;
    size_t* stuc_layers;
    double* m_params;
    int m_regularizer;
    std::vector<MatrixXr*>* m_weights;
    std::vector<VectorXr*>* m_biases;
};

#endif
