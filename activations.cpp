#include "global_def.hpp"
#include "activations.hpp"

void activations::softplus(MatrixXr* _vx){
    (*_vx) = (*_vx).array().exp() + 1;
    (*_vx) = (*_vx).array().log();
}

void activations::sigmoid(MatrixXr* _vx){
    (*_vx) = (*_vx).array().exp() / ((*_vx).array().exp() + 1);
}

void activations::softmax(MatrixXr* _vx){
    (*_vx) = (*_vx).array().exp();
    double exp_sum = (*_vx).sum();
    (*_vx) = (*_vx) / exp_sum;
}

// _D: row vector
void activations::softplus_1th_derivative(Tuple _pF, Tuple _F, MatrixXr* _pX
    , MatrixXr* _X, MatrixXr* _D, size_t N) {
    MatrixXr t_del = (*_F._w) * (*_X) + (*_F._b);
    MatrixXr t_sig = t_del.array().exp();
    t_del = t_sig.array() / (t_sig.array() + 1).array();
    t_del = ((*_D).array() * t_del.transpose().array()).transpose();
    (*_pF._w) += (t_del * (*_X).transpose()) / N;
    (*_pF._b) += t_del / N;
    (*_pX) = t_del.transpose() * (*_F._w);
}

// _pX: return row vector
void activations::loss_1th_derivative(Tuple _pF, Tuple _F, MatrixXr* _pX, MatrixXr* _X
    , MatrixXr* _Y, size_t N) {
    MatrixXr p = (*_F._w) * (*_X) + (*_F._b);
    softmax(&p);
    // d_loss/d_xi = (pi - yi)
    MatrixXr dl_dx = (p - *_Y);
    (*_pF._w) += (dl_dx * (*_X).transpose()) / N;
    (*_pF._b) += dl_dx / N;
    (*_pX) = dl_dx.transpose() * (*_F._w);
}
