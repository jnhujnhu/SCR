#ifndef ACTIVATIONSHPP
#define ACTIVATIONSHPP
#include "global_def.hpp"

using namespace global_def;
namespace activations {
    void softplus(MatrixXr* _vx);
    // softplus_1th_derivative = sigmoid
    void sigmoid(MatrixXr* _vx);
    void softmax(MatrixXr* _vx);
    void softmax_1th_derivative(MatrixXr* _vx);
    void softplus_1th_backpropagation(Tuple _pF, Tuple _F, MatrixXr* _pX, MatrixXr* _X
        , MatrixXr* _D, size_t N, bool is_input_layer);
    void softplus_2th_hessian_vector_bp(Tuple _hvF, Tuple _F, Tuple _V, MatrixXr* _pX
        , MatrixXr* _hvX, MatrixXr* _X, MatrixXr* _RX, MatrixXr* _D, MatrixXr* _hvD
        , size_t N, bool is_input_layer);
    void loss_1th_backpropagation(Tuple _pF, Tuple _F, MatrixXr* _pX, MatrixXr* _X
        , MatrixXr* _Y, size_t N);
    void loss_2th_hessian_vector_bp(Tuple _hvF, Tuple _F, Tuple _V, MatrixXr* _pX
        , MatrixXr* _hvX, MatrixXr* _X, MatrixXr* _RX, MatrixXr* _Y, size_t N);
}

#endif
