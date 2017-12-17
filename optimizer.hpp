#ifndef OPTIMIZERHPP
#define OPTIMIZERHPP
#include "global_def.hpp"
#include "DNN.hpp"

using namespace global_def;
namespace optimizer {
    class outputs {
    public:
        outputs(std::vector<double>* losses, std::vector<double>* accuracies):
            _losses(losses), _accuracies(accuracies) {}
        std::vector<double>* _losses;
        std::vector<double>* _accuracies;
    };
    // Sampling without replacement
    Batch random_batch_generator(Batch full_batch, size_t batch_size
        , bool is_autoencoder);
    bool standard_trace(DNN* dnn, size_t i, Batch train_batch, Batch test_batch
        , std::vector<double>* loss_shots, std::vector<double>* acc_shots);
    outputs SGD(DNN* dnn, Batch train_batch, Batch test_batch, size_t n_batch_size
        , size_t n_iteraions, size_t n_save_interval, double step_size
        , double decay, bool f_save);
    outputs Adam(DNN* dnn, Batch train_batch, Batch test_batch, size_t n_batch_size
        , size_t n_iteraions, size_t n_save_interval, double step_size, double beta1
        , double beta2, double epsilon, bool f_save);
    outputs AdaGrad(DNN* dnn, Batch train_batch, Batch test_batch, size_t n_batch_size
        , size_t n_iteraions, size_t n_save_interval, double step_size, double epsilon
        , bool f_save);
    outputs SCR(DNN* dnn, Batch train_batch, Batch test_batch, size_t g_batch_size
        , size_t hv_batch_size, size_t n_iteraions, size_t sub_iterations
        , size_t n_save_interval, size_t petb_interval, double eta, double rho
        , double sigma, bool f_save);
}

#endif
