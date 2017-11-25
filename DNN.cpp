#include "DNN.hpp"
#include "activations.hpp"

DNN::DNN(size_t i_n_layers, size_t* i_stuc_layers, size_t n_params, double* params
    , double std_dev, int regularizer) {
    n_layers = i_n_layers + 2;
    stuc_layers = new size_t[n_layers];
    stuc_layers[0] = DIM;
    stuc_layers[n_layers - 1] = CLASS;
    memcpy(&stuc_layers[1], i_stuc_layers, i_n_layers * sizeof(size_t));

    m_params = new double[n_params];
    memcpy(m_params, params, n_params * sizeof(double));
    m_regularizer = regularizer;

    m_weights = new std::vector<MatrixXr*>;
    m_biases = new std::vector<VectorXr*>;
    for(size_t i = 0; i < n_layers - 1; i ++) {
        MatrixXr* t_weights = new MatrixXr(stuc_layers[i + 1], stuc_layers[i]);
        (*t_weights).setRandom(stuc_layers[i + 1], stuc_layers[i]);
        (*t_weights) = (*t_weights) * std_dev;
        m_weights->push_back(t_weights);
        VectorXr* t_biases = new VectorXr(stuc_layers[i + 1]);
        (*t_biases).setRandom(stuc_layers[i + 1]);
        (*t_biases) = (*t_biases) * std_dev;
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

void DNN::initialize(double std_dev) {
    for(size_t i = 0; i < n_layers - 1; i ++) {
        (*m_weights)[i]->setRandom(stuc_layers[i + 1], stuc_layers[i]);
        *(*m_weights)[i] = *(*m_weights)[i] * std_dev;
        (*m_biases)[i]->setRandom(stuc_layers[i + 1]);
        *(*m_biases)[i] = *(*m_biases)[i] * std_dev;
    }
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
}

void DNN::update_parameters(std::vector<Tuple> tuples) {
    for(size_t i = 0; i < n_layers - 1; i ++) {
        *(*m_weights)[i] += *tuples[i]._w;
        *(*m_biases)[i] += *tuples[i]._b;
    }
}

double DNN::zero_oracle(Batch batch) {
    MatrixXr* X = batch._X;
    MatrixXr* Y = batch._Y;
    size_t N = batch._n;

    double loss = 0;
    for(size_t i = 0; i < N; i ++) {
        MatrixXr temp = (*X).row(i).transpose();
        for (size_t j = 0; j < n_layers - 1; j ++) {
            temp = *(*m_weights)[j] * temp + *(*m_biases)[j];
            if(j < n_layers - 2)
                activations::softplus(&temp);
            else
                activations::softmax(&temp);
        }
        loss += - ((*Y).row(i).transpose().array() * (temp.array().log()).array()).sum() / N;
    }
    return loss;
}

std::vector<Tuple> DNN::first_oracle(Batch batch) {
    MatrixXr* X = batch._X;
    MatrixXr* Y = batch._Y;
    size_t N = batch._n;

    std::vector<Tuple> result;
    for(size_t i = 0; i < n_layers - 1; i ++) {
        MatrixXr* t_weights = new MatrixXr(stuc_layers[i + 1], stuc_layers[i]);
        (*t_weights).setZero(stuc_layers[i + 1], stuc_layers[i]);
        VectorXr* t_biases = new VectorXr(stuc_layers[i + 1]);
        (*t_biases).setZero(stuc_layers[i + 1]);
        Tuple tuple(t_weights, t_biases);
        result.push_back(tuple);
    }
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
        for(int j = n_layers - 2; j >= 0; j --) {
            if(j == int(n_layers) - 2) {
                MatrixXr _Y = (*Y).row(i).transpose();
                MatrixXr _pX(1, stuc_layers[j]);
                Tuple t_F((*m_weights)[j], (*m_biases)[j]);
                activations::loss_1th_derivative(result[j], t_F, &_pX, &_X[j]
                    , &_Y, N);
                _D = _pX;
            }
            else {
                MatrixXr _pX(1, stuc_layers[j]);
                Tuple t_F((*m_weights)[j], (*m_biases)[j]);
                activations::softplus_1th_derivative(result[j], t_F, &_pX, &_X[j]
                    , &_D, N);
                _D = _pX;
            }
        }
    }
    return result;
}

// std::vector<Tuple> DNN::hessian_vector_oracle(Batch batch, MatrixXr* V) {
//
// }
