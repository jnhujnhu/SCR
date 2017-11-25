#include "optimizer.hpp"

Batch optimizer::random_batch_generator(MatrixXr* X, MatrixXr* Y, size_t batch_size) {
    // Random Generator
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::vector<int> indexes;
    MatrixXr* _X = new MatrixXr(batch_size, DIM);
    MatrixXr* _Y = new MatrixXr(batch_size, CLASS);
    for(size_t i = 0; i < NF; i ++)
        indexes.push_back(i);
    for(size_t i = 0;i < batch_size; i ++) {
        std::uniform_int_distribution<int> distribution(0, NF - 1 - i);
        int k = distribution(generator);
        _X->row(i) = X->row(indexes[k]);
        _Y->row(i) = Y->row(indexes[k]);
        indexes.erase(indexes.begin() + k);
    }
    Batch batch(_X, _Y, batch_size);
    return batch;
};

std::vector<double>* optimizer::SGD(DNN* dnn, MatrixXr* X, MatrixXr* Y, size_t n_batch_size
    , size_t n_iteraions, size_t n_save_interval, double step_size, bool f_save) {
    std::vector<double>* loss_shots = new std::vector<double>;
    Batch full_batch(X, Y, NF);
    if(f_save)
        loss_shots->push_back(dnn->zero_oracle(full_batch));
    for(size_t i = 0; i < n_iteraions; i ++) {
        Batch minibatch = random_batch_generator(X, Y, n_batch_size);
        std::vector<Tuple> tuples = dnn->first_oracle(minibatch);
        for(auto tuple : tuples)
            tuple *= -step_size;
        dnn->update_parameters(tuples);
        if(f_save && !(i % n_save_interval)) {
            loss_shots->push_back(dnn->zero_oracle(full_batch));
        }
        // Clean up temp memory
        for(auto tuple : tuples)
            tuple.clean_up();
        minibatch.clean_up();
    }
    return loss_shots;
}
