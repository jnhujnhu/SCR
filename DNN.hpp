#ifndef DNN_HPP
#define DNN_HPP
#include <stdio.h>
#include "global_def.hpp"

using namespace global_def;
class DNN {
public:
    DNN(size_t i_n_layers, size_t* i_stuc_layers, size_t n_params, double* params
        , double std_dev = 1.0, size_t initializer = I_UNIFORM
        , int regularizer = L2);
    ~DNN();
    void print_all();
    // passing increments
    void update_parameters(std::vector<Tuple> tuples);
    double get_accuracy(Batch test_batch);
    double zero_oracle(Batch batch);
    std::vector<Tuple> first_oracle(Batch batch);
    std::vector<Tuple> hessian_vector_oracle(Batch batch, std::vector<Tuple> V);
    std::vector<Tuple> hessian_vector_approxiamate_oracle(Batch batch
        , std::vector<Tuple> grad, std::vector<Tuple> V);

    std::vector<Tuple> get_zero_tuples();
    std::vector<Tuple> get_ones_tuples();
    // Generate perturbing Tuples
    std::vector<Tuple> get_perturb_tuples();
    size_t get_n_layers();
private:
    template<typename Derived>
    void initialize(Eigen::PlainObjectBase<Derived>* _mx, double std_dev
        , size_t method = I_UNIFORM);
    static double gaussian_unary(double dummy);
    size_t n_layers;
    size_t* stuc_layers;
    double* m_params;
    int m_regularizer;
    std::vector<MatrixXr*>* m_weights;
    std::vector<VectorXr*>* m_biases;
};

#endif
