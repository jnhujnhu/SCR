#ifndef OPTIMIZERHPP
#define OPTIMIZERHPP
#include "global_def.hpp"
#include "DNN.hpp"

using namespace global_def;
namespace optimizer {
    // Sampling without replacement
    Batch random_batch_generator(MatrixXr* X, MatrixXr* Y, size_t batch_size);
    std::vector<double>* SGD(DNN* dnn, MatrixXr* X, MatrixXr* Y, size_t n_batch_size
        , size_t n_iteraions, size_t n_save_interval, double step_size, bool f_save);
    std::vector<double>* Adam(DNN* dnn, MatrixXr* X, MatrixXr* Y, size_t n_batch_size
        , size_t n_iteraions, size_t n_save_interval, double step_size, double beta1
        , double beta2, double epsilon, bool f_save);
}

#endif
