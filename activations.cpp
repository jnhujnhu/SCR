#include "global_def.hpp"
#include "activations.hpp"

void activations::softplus(MatrixXr* _vx){
    (*_vx) = (*_vx).array().exp() + 1;
    (*_vx) = (*_vx).array().log();
}

void activations::sigmoid(MatrixXr* _vx){
    (*_vx) = 1.0 / ((-*_vx).array().exp() + 1);
}

void activations::softmax(MatrixXr* _vx){
    (*_vx) = (*_vx).array().exp();
    double exp_sum = (*_vx).sum();
    (*_vx) = (*_vx) / exp_sum;
}

// _D: row vector
void activations::softplus_1th_backpropagation(Tuple _pF, Tuple _F, MatrixXr* _pX
    , MatrixXr* _X, MatrixXr* _D, size_t Const, bool is_input_layer) {
    MatrixXr t_del = (*_F._w) * (*_X) + (*_F._b);
    MatrixXr t_sig = t_del.array().exp();
    t_sig = t_sig.array() / (t_sig.array() + 1).array();
    t_del = ((*_D).transpose().array() * t_sig.array());
    (*_pF._w) += (t_del * (*_X).transpose()) / Const;
    (*_pF._b) += t_del / Const;
    (*_pX) = t_del.transpose() * (*_F._w);
}

// _D: row vector
void activations::softplus_2th_hessian_vector_bp(Tuple _hvF, Tuple _F, Tuple _V
    , MatrixXr* _pX, MatrixXr* _hvX, MatrixXr* _X, MatrixXr* _RX, MatrixXr* _D
    , MatrixXr* _hvD, size_t Const, bool is_input_layer) {
    MatrixXr dl_dy = (*_F._w) * (*_X) + (*_F._b);
    MatrixXr v = (*_V._w) * (*_X) + (*_V._b) + (*_F._w) * (*_RX);
    MatrixXr df_dy = dl_dy.array().exp();

    df_dy = df_dy.array() / (df_dy.array() + 1).array();
    dl_dy = ((*_D).transpose().array() * df_dy.array());

    if(!is_input_layer)
        (*_pX) = dl_dy.transpose() * (*_F._w);

    MatrixXr R_df_dy = df_dy.array() * (1 - df_dy.array()) * v.array();

    MatrixXr Rdl_dy = (*_hvD).transpose().array() * df_dy.array();
    MatrixXr D_Rfy = (*_D).transpose().array() * R_df_dy.array();
    MatrixXr D_fy = (*_D).transpose().array() * df_dy.array();

    MatrixXr D_Rff = D_Rfy * (*_X).transpose() + D_fy * (*_RX).transpose();

    (*_hvF._w) += (Rdl_dy * (*_X).transpose() + D_Rff) / Const;
    (*_hvF._b) += (Rdl_dy + ((*_D).transpose().array() * R_df_dy.array()).matrix()) / Const;

    if(!is_input_layer) {
        MatrixXr D_Rfx = D_Rfy.transpose()  * (*_F._w) + D_fy.transpose() * (*_V._w);
        (*_hvX) = (Rdl_dy.transpose() * (*_F._w)) + D_Rfx;
    }
}

// _pX: return row vector
void activations::loss_1th_backpropagation(Tuple _pF, Tuple _F, MatrixXr* _pX
    , MatrixXr* _X, MatrixXr* _Y, size_t Const) {
    MatrixXr p = (*_F._w) * (*_X) + (*_F._b);
    softmax(&p);
    // d_loss/d_yi = (pi - Yi)
    MatrixXr dl_dy = p - *_Y;
    (*_pF._w) += (dl_dy * (*_X).transpose()) / Const;
    (*_pF._b) += dl_dy / Const;
    (*_pX) = dl_dy.transpose() * (*_F._w);
}

// _pX: return row vector
void activations::loss_2th_hessian_vector_bp(Tuple _hvF, Tuple _F, Tuple _V
    , MatrixXr* _pX, MatrixXr* _hvX, MatrixXr* _X, MatrixXr* _RX, MatrixXr* _Y
    , size_t Const) {
    MatrixXr p = (*_F._w) * (*_X) + (*_F._b);
    MatrixXr R_dl_dy = (*_V._w) * (*_X) + (*_V._b) + (*_F._w) * (*_RX);
    softmax(&p);
    MatrixXr t_1 = p.array() * R_dl_dy.array();
    R_dl_dy = t_1 - p * (p.transpose() * R_dl_dy);
    MatrixXr dl_dy = p - *_Y;
    (*_hvF._w) += (R_dl_dy * (*_X).transpose() + dl_dy * (*_RX).transpose()) / Const;
    (*_hvF._b) += R_dl_dy / Const;
    (*_hvX) = R_dl_dy.transpose() * (*_F._w) + dl_dy.transpose() * (*_V._w);
    (*_pX) = dl_dy.transpose() * (*_F._w);
}

// _pX: return row vector
void activations::l2loss_1th_backpropagation(Tuple _pF, Tuple _F, MatrixXr* _pX
    , MatrixXr* _X, MatrixXr* _Y, size_t Const) {
    MatrixXr p = (*_F._w) * (*_X) + (*_F._b);
    sigmoid(&p);
    // d_loss/d_yi = (pi - Yi) * pi * (1 - pi)
    MatrixXr dl_dy = (p - *_Y).array() * p.array() * (1 - p.array());
    (*_pF._w) += (dl_dy * (*_X).transpose()) / Const;
    (*_pF._b) += dl_dy / Const;
    (*_pX) = dl_dy.transpose() * (*_F._w);
}

// _pX: return row vector
void activations::l2loss_2th_hessian_vector_bp(Tuple _hvF, Tuple _F, Tuple _V
    , MatrixXr* _pX, MatrixXr* _hvX, MatrixXr* _X, MatrixXr* _RX, MatrixXr* _Y
    , size_t Const) {
    MatrixXr p = (*_F._w) * (*_X) + (*_F._b);
    MatrixXr R_dl_dy = (*_V._w) * (*_X) + (*_V._b) + (*_F._w) * (*_RX);
    MatrixXr e_p = (-p).array().exp();
    MatrixXr e_p_2 = e_p.array().pow(2);
    R_dl_dy = (2 * e_p_2 - e_p + ((*_Y).array() * (e_p_2.array() * e_p.array()
        - e_p.array())).matrix()).array() / (e_p.array() + 1).pow(4) * R_dl_dy.array();
    sigmoid(&p);
    MatrixXr dl_dy = (p - *_Y).array() * p.array() * (1 - p.array());
    (*_hvF._w) += (R_dl_dy * (*_X).transpose() + dl_dy * (*_RX).transpose()) / Const;
    (*_hvF._b) += R_dl_dy / Const;
    (*_hvX) = R_dl_dy.transpose() * (*_F._w) + dl_dy.transpose() * (*_V._w);
    (*_pX) = dl_dy.transpose() * (*_F._w);
}
