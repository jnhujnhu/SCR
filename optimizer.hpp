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
    Batch random_batch_generator(Batch full_batch, size_t batch_size);
    outputs SGD(DNN* dnn, Batch train_batch, Batch test_batch, size_t n_batch_size
        , size_t n_iteraions, size_t n_save_interval, double step_size, bool f_save);
    outputs Adam(DNN* dnn, Batch train_batch, Batch test_batch, size_t n_batch_size
        , size_t n_iteraions, size_t n_save_interval, double step_size, double beta1
        , double beta2, double epsilon, bool f_save);
}

#endif
