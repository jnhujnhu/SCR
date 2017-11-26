#include "optimizer.hpp"

Batch optimizer::random_batch_generator(Batch full_batch, size_t batch_size) {
    // Random Generator
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::vector<int> indexes;
    MatrixXr* _X = new MatrixXr(batch_size, DIM);
    MatrixXr* _Y = new MatrixXr(batch_size, CLASS);
    for(size_t i = 0; i < full_batch._n; i ++)
        indexes.push_back(i);
    for(size_t i = 0;i < batch_size; i ++) {
        std::uniform_int_distribution<int> distribution(0, full_batch._n - 1 - i);
        int k = distribution(generator);
        _X->row(i) = full_batch._X->row(indexes[k]);
        _Y->row(i) = full_batch._Y->row(indexes[k]);
        indexes.erase(indexes.begin() + k);
    }
    Batch batch(_X, _Y, batch_size);
    return batch;
};

optimizer::outputs optimizer::SGD(DNN* dnn, Batch train_batch, Batch test_batch
    , size_t n_batch_size, size_t n_iteraions, size_t n_save_interval, double step_size
    , bool f_save) {
    std::vector<double>* loss_shots = new std::vector<double>;
    std::vector<double>* acc_shots = new std::vector<double>;
    if(f_save) {
        loss_shots->push_back(dnn->zero_oracle(train_batch));
        acc_shots->push_back(dnn->get_accuracy(test_batch));
    }
    for(size_t i = 0; i < n_iteraions; i ++) {
        Batch minibatch = random_batch_generator(train_batch, n_batch_size);
        std::vector<Tuple> tuples = dnn->first_oracle(minibatch);
        for(auto tuple : tuples)
            tuple *= -step_size;
        dnn->update_parameters(tuples);
        if(f_save && !(i % n_save_interval)) {
            double loss = dnn->zero_oracle(train_batch);
            double acc = dnn->get_accuracy(test_batch);
            loss_shots->push_back(loss);
            acc_shots->push_back(acc);
            std::cout.precision(13);
            std::cout << "Iteration " << i << " with loss = " << loss
                << " acc = " << acc << std::endl;
        }
        // Clean up temp memory
        for(auto tuple : tuples)
            tuple.clean_up();
        minibatch.clean_up();
    }
    optimizer::outputs outs(loss_shots, acc_shots);
    return outs;
}
