#include "DNN.hpp"
#include "activations.hpp"
#include <time.h>

DNN::DNN(size_t i_n_layers, size_t* i_stuc_layers, size_t n_params, double* params
    , double std_dev, size_t initializer, bool is_autoencoder, int regularizer) {

    this->is_autoencoder = is_autoencoder;
    if(is_autoencoder) {
        n_layers = 2 * i_n_layers + 3;
        stuc_layers = new size_t[n_layers];
        // Symmetric Encoder & Decoder
        stuc_layers[0] = DIM;
        stuc_layers[i_n_layers + 1] = CLASS;
        stuc_layers[n_layers - 1] = DIM;
        memcpy(&stuc_layers[1], i_stuc_layers, i_n_layers * sizeof(size_t));
        for(size_t i = 0; i < i_n_layers; i ++)
            stuc_layers[i_n_layers + 2 + i] = i_stuc_layers[i_n_layers - 1 - i];
    }
    else {
        n_layers = i_n_layers + 2;
        stuc_layers = new size_t[n_layers];
        stuc_layers[0] = DIM;
        stuc_layers[n_layers - 1] = CLASS;
        memcpy(&stuc_layers[1], i_stuc_layers, i_n_layers * sizeof(size_t));
    }

    m_params = new double[n_params];
    memcpy(m_params, params, n_params * sizeof(double));
    m_regularizer = regularizer;

    srand((unsigned int) time(0));
    m_weights = new std::vector<MatrixXr*>;
    m_biases = new std::vector<VectorXr*>;
    for(size_t i = 0; i < n_layers - 1; i ++) {
        MatrixXr* t_weights = new MatrixXr(stuc_layers[i + 1], stuc_layers[i]);
        initialize(t_weights, std_dev, initializer);
        m_weights->push_back(t_weights);
        VectorXr* t_biases = new VectorXr(stuc_layers[i + 1]);
        initialize(t_biases, std_dev, initializer);
        m_biases->push_back(t_biases);
    }
}

DNN::~DNN() {
    delete[] m_params;
    delete[] stuc_layers;
    for(auto weights : (*m_weights)) {
        delete weights;
    }
    for(auto biases : (*m_biases)) {
        delete biases;
    }
    delete m_weights;
    delete m_biases;
}

template<typename Derived>
void DNN::initialize(Eigen::PlainObjectBase<Derived>* _mx, double std_dev
    , size_t method) {
    switch(method) {
        case I_UNIFORM:
            _mx->setRandom(_mx->rows(), _mx->cols());
            *_mx = *_mx * std_dev;
            break;
        case I_GAUSSIAN:{
            std::random_device rd;
            std::default_random_engine generator(rd());
            std::normal_distribution<double> distribution(0, std_dev);
            for(size_t i = 0; i < _mx->rows(); i ++)
                for(size_t j = 0; j < _mx->cols(); j ++)
                    (*_mx)(i, j) = distribution(generator);
            break;
        }
        case I_ZERO:
            _mx->setZero(_mx->rows(), _mx->cols());
            break;
    }
}

double DNN::gaussian_unary(double dummy) {
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::normal_distribution<double> distribution(0, 1);
    return distribution(generator);
}

void DNN::print_all() {
    for(size_t i = 0; i < n_layers - 1; i ++) {
        std::cout << "========  LAYER " << i + 1 << "  ========" << std::endl;
        std::cout << "Weights(" << (*m_weights)[i]->rows()
            <<"x"<< (*m_weights)[i]->cols() <<"):\n";
        std::cout << *(*m_weights)[i] << std::endl;
        std::cout << "Biases(" << (*m_biases)[i]->rows()
            <<"x1):\n";
        std::cout << *(*m_biases)[i] << std::endl;
    }
    std::cout << "========  LAYER " << n_layers << "  ========" << std::endl;
}

void DNN::update_parameters(std::vector<Tuple> tuples) {
    for(size_t i = 0; i < n_layers - 1; i ++) {
        *(*m_weights)[i] += *tuples[i]._w;
        *(*m_biases)[i] += *tuples[i]._b;
    }
}

std::vector<Tuple> DNN::get_zero_tuples() {
    std::vector<Tuple> zero_tuples;
    for(size_t i = 0; i < n_layers - 1; i ++) {
        MatrixXr* t_weights = new MatrixXr(stuc_layers[i + 1], stuc_layers[i]);
        (*t_weights).setZero(stuc_layers[i + 1], stuc_layers[i]);
        VectorXr* t_biases = new VectorXr(stuc_layers[i + 1]);
        (*t_biases).setZero(stuc_layers[i + 1]);
        Tuple tuple(t_weights, t_biases);
        zero_tuples.push_back(tuple);
    }
    return zero_tuples;
}

std::vector<Tuple> DNN::get_ones_tuples() {
    std::vector<Tuple> zero_tuples;
    for(size_t i = 0; i < n_layers - 1; i ++) {
        MatrixXr* t_weights = new MatrixXr(stuc_layers[i + 1], stuc_layers[i]);
        (*t_weights).setOnes(stuc_layers[i + 1], stuc_layers[i]);
        VectorXr* t_biases = new VectorXr(stuc_layers[i + 1]);
        (*t_biases).setOnes(stuc_layers[i + 1]);
        Tuple tuple(t_weights, t_biases);
        zero_tuples.push_back(tuple);
    }
    return zero_tuples;
}

std::vector<Tuple> DNN::get_perturb_tuples() {
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::normal_distribution<double> distribution(0, 1);
    std::vector<Tuple> petb_tuples;

    for(size_t i = 0; i < n_layers - 1; i ++) {
        MatrixXr* t_weights = new MatrixXr(stuc_layers[i + 1], stuc_layers[i]);
        t_weights->unaryExpr(std::ptr_fun(gaussian_unary));
        t_weights->normalize();
        VectorXr* t_biases = new VectorXr(stuc_layers[i + 1]);
        t_biases->unaryExpr(std::ptr_fun(gaussian_unary));
        t_biases->normalize();
        Tuple tuple(t_weights, t_biases);
        petb_tuples.push_back(tuple);
    }
    return petb_tuples;
}

size_t DNN::get_n_layers() {
    return this->n_layers;
}

bool DNN::isAutoencoder() {
    return this->is_autoencoder;
}

double DNN::zero_oracle(Batch batch) {
    MatrixXr* X = batch._X;
    MatrixXr* Y = batch._Y;
    size_t N = batch._n;

    double loss = 0, regularizer = 0;
    for(size_t i = 0; i < N; i ++) {
        MatrixXr temp = (*X).row(i).transpose();
        for (size_t j = 0; j < n_layers - 1; j ++) {
            temp = *(*m_weights)[j] * temp + *(*m_biases)[j];
            if(j < n_layers - 2)
                activations::softplus(&temp);
            else {
                if(is_autoencoder)
                    activations::sigmoid(&temp);
                else
                    activations::softmax(&temp);
            }
            // Add L2 Regularizer
            if(i == 0) {
                regularizer += (*(*m_weights)[j]).squaredNorm()
                        + (*(*m_biases)[j]).squaredNorm();
            }
        }
        // Pixel-wise L2 Loss
        if(is_autoencoder)
            loss += ((*Y).row(i).transpose() - temp).squaredNorm() / (2.0 * DIM * N);
        else
            loss += - ((*Y).row(i).transpose().array() * (temp.array().log()).array()).mean() / N;
    }
    return loss + m_params[0] / 2 * regularizer;
}

std::vector<Tuple> DNN::first_oracle(Batch batch) {
    MatrixXr* X = batch._X;
    MatrixXr* Y = batch._Y;
    size_t N = batch._n;

    std::vector<Tuple> result = get_zero_tuples();
    for(size_t i = 0; i < N; i ++) {
        MatrixXr temp = (*X).row(i).transpose();
        std::vector<MatrixXr> _X;
        _X.push_back(temp);
        // feed forward
        for (size_t j = 0; j < n_layers - 2; j ++) {
            temp = *(*m_weights)[j] * temp + *(*m_biases)[j];
            activations::softplus(&temp);
            _X.push_back(temp);
        }
        // back propagation
        MatrixXr _D;
        size_t avg_const = (is_autoencoder) ? N * DIM : N * CLASS;
        for(int j = n_layers - 2; j >= 0; j --) {
            if(j == int(n_layers) - 2) {
                MatrixXr _Y = (*Y).row(i).transpose();
                MatrixXr _pX(1, stuc_layers[j]);
                Tuple t_F((*m_weights)[j], (*m_biases)[j]);
                if(is_autoencoder)
                    activations::l2loss_1th_backpropagation(result[j], t_F, &_pX, &_X[j]
                        , &_Y, avg_const);
                else
                    activations::loss_1th_backpropagation(result[j], t_F, &_pX, &_X[j]
                        , &_Y, avg_const);
                _D = _pX;
            }
            else {
                MatrixXr _pX(1, stuc_layers[j]);
                Tuple t_F((*m_weights)[j], (*m_biases)[j]);
                activations::softplus_1th_backpropagation(result[j], t_F, &_pX, &_X[j]
                    , &_D, avg_const, (j == 0) ? true : false);
                _D = _pX;
            }
        }
    }
    // Add L2 Regularizer
    for (size_t j = 0; j < n_layers - 1; j ++) {
        Tuple t_F((*m_weights)[j], (*m_biases)[j]);
        result[j](t_F, m_params[0]);
    }
    return result;
}

std::vector<Tuple> DNN::hessian_vector_approxiamate_oracle(Batch batch
    , std::vector<Tuple> grad, std::vector<Tuple> V) {
    double r = 0.01;
    for(size_t i = 0; i < n_layers - 1; i ++) {
        *(*m_weights)[i] += r * (*V[i]._w);
        *(*m_biases)[i] += r * (*V[i]._b);
    }
    std::vector<Tuple> result = first_oracle(batch);
    for(size_t i = 0; i < n_layers - 1; i ++) {
        result[i](grad[i], -1);
        result[i] *= 1.0 / r;
        // Restore
        *(*m_weights)[i] += -r * (*V[i]._w);
        *(*m_biases)[i] += -r * (*V[i]._b);
    }
    return result;
}

// Using R{} Technique
std::vector<Tuple> DNN::hessian_vector_oracle(Batch batch, std::vector<Tuple> V) {
    MatrixXr* X = batch._X;
    MatrixXr* Y = batch._Y;
    size_t N = batch._n;

    std::vector<Tuple> result = get_zero_tuples();
    for(size_t i = 0; i < N; i ++) {
        MatrixXr temp_X = (*X).row(i).transpose();
        MatrixXr temp_RX = temp_X;
        temp_RX.setZero();
        std::vector<MatrixXr> _X;
        std::vector<MatrixXr> _RX;
        _X.push_back(temp_X);
        _RX.push_back(temp_RX);
        // HV feed forward
        for (size_t j = 0; j < n_layers - 2; j ++) {
            MatrixXr prev_temp_X = temp_X, prev_temp_RX = temp_RX;
            temp_X = *(*m_weights)[j] * temp_X + *(*m_biases)[j];
            temp_RX = temp_X;
            activations::softplus(&temp_X);
            // softplus_1th_derivative
            activations::sigmoid(&temp_RX);
            temp_RX = temp_RX.array() * (*V[j]._w * prev_temp_X
                              + *(*m_weights)[j] * prev_temp_RX
                              + *V[j]._b).array();
            _X.push_back(temp_X);
            _RX.push_back(temp_RX);
        }
        // HV back propagation
        MatrixXr _D, _hvD;
        size_t avg_const = (is_autoencoder) ? N * DIM : N * CLASS;
        for(int j = n_layers - 2; j >= 0; j --) {
            if(j == int(n_layers) - 2) {
                MatrixXr _Y = (*Y).row(i).transpose();
                MatrixXr _pX(1, stuc_layers[j]);
                MatrixXr _hvX(1, stuc_layers[j]);
                Tuple t_F((*m_weights)[j], (*m_biases)[j]);
                if(is_autoencoder)
                    activations::l2loss_2th_hessian_vector_bp(result[j], t_F, V[j], &_pX
                        , &_hvX, &_X[j], &_RX[j], &_Y, avg_const);
                else
                    activations::loss_2th_hessian_vector_bp(result[j], t_F, V[j], &_pX
                        , &_hvX, &_X[j], &_RX[j], &_Y, avg_const);
                _D = _pX;
                _hvD = _hvX;
            }
            else {
                MatrixXr _pX(1, stuc_layers[j]);
                MatrixXr _hvX(1, stuc_layers[j]);
                Tuple t_F((*m_weights)[j], (*m_biases)[j]);
                activations::softplus_2th_hessian_vector_bp(result[j], t_F, V[j]
                    , &_pX, &_hvX, &_X[j], &_RX[j], &_D, &_hvD, avg_const
                    , (j == 0) ? true : false);
                _D = _pX;
                _hvD = _hvX;
            }
        }
    }
    // Add L2 Regularizer
    for (size_t j = 0; j < n_layers - 1; j ++)
        result[j](V[j], m_params[0]);
    return result;
}

double DNN::get_accuracy(Batch test_batch) {
    MatrixXr* X = test_batch._X;
    MatrixXr* Y = test_batch._Y;
    size_t N = test_batch._n;

    double accuracy = 0.0;
    for(size_t i = 0; i < N; i ++) {
        MatrixXr temp = (*X).row(i).transpose();
        for (size_t j = 0; j < n_layers - 1; j ++) {
            temp = *(*m_weights)[j] * temp + *(*m_biases)[j];
            if(j < n_layers - 2)
                activations::softplus(&temp);
            else
                activations::softmax(&temp);
        }
        int argmax_Y, argmax_temp;
        (*Y).row(i).maxCoeff(&argmax_Y);
        temp.col(0).maxCoeff(&argmax_temp);
        if(argmax_temp == argmax_Y)
            accuracy += 1.0 / N;
    }
    return accuracy;
}
