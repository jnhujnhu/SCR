#ifndef ACTIVATIONSHPP
#define ACTIVATIONSHPP
#include "global_def.hpp"

using namespace global_def;
namespace activations {
    void softplus(MatrixXr* _vx);
    void sigmoid(MatrixXr* _vx);
    void softmax(MatrixXr* _vx);
    void softplus_1th_derivative(Tuple _pF, Tuple _F, MatrixXr* _pX, MatrixXr* _X
        , MatrixXr* _D, size_t N);
    void softplus_2th_hessian_vector(Tuple _hvF, Tuple _F, Tuple _V, MatrixXr* _pX
        , MatrixXr* _hvX, MatrixXr* _X, MatrixXr* _D, MatrixXr* _hvD, size_t N);
    void loss_1th_derivative(Tuple _pF, Tuple _F, MatrixXr* _pX, MatrixXr* _X
        , MatrixXr* _Y, size_t N);
    void loss_2th_hessian_vector(Tuple _hvF, Tuple _F, Tuple _V, MatrixXr* _pX
        , MatrixXr* _hvX, MatrixXr* _X, MatrixXr* _Y, size_t N);
}

#endif
