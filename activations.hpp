#ifndef ACTIVATIONSHPP
#define ACTIVATIONSHPP
#include "global_def.hpp"

using namespace global_def;
namespace activations {
    void softplus(MatrixXr* _vx);
    void sigmoid(MatrixXr* _vx);
    void softmax(MatrixXr* _vx);
    void softmax_1th_derivative(MatrixXr* _vx);
    void softplus_1th_backpropagation(Tuple _pF, Tuple _F, MatrixXr* _pX, MatrixXr* _X
        , MatrixXr* _D, size_t Const, bool is_input_layer);
    void softplus_2th_hessian_vector_bp(Tuple _hvF, Tuple _F, Tuple _V, MatrixXr* _pX
        , MatrixXr* _hvX, MatrixXr* _X, MatrixXr* _RX, MatrixXr* _D, MatrixXr* _hvD
        , size_t Const, bool is_input_layer);
    // Combined with softmax for Classification
    void loss_1th_backpropagation(Tuple _pF, Tuple _F, MatrixXr* _pX, MatrixXr* _X
        , MatrixXr* _Y, size_t Const);
    void loss_2th_hessian_vector_bp(Tuple _hvF, Tuple _F, Tuple _V, MatrixXr* _pX
        , MatrixXr* _hvX, MatrixXr* _X, MatrixXr* _RX, MatrixXr* _Y, size_t Const);
    // Combined with sigmoid for Autoencoder
    void l2loss_1th_backpropagation(Tuple _pF, Tuple _F, MatrixXr* _pX, MatrixXr* _X
        , MatrixXr* _Y, size_t Const);
    void l2loss_2th_hessian_vector_bp(Tuple _hvF, Tuple _F, Tuple _V, MatrixXr* _pX
        , MatrixXr* _hvX, MatrixXr* _X, MatrixXr* _RX, MatrixXr* _Y, size_t Const);
}

#endif
