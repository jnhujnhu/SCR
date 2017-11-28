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
        std::vector<Tuple> grad = dnn->first_oracle(minibatch);
        for(auto tuple : grad)
            tuple *= -step_size;
        dnn->update_parameters(grad);
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
        for(auto tuple : grad)
            tuple.clean_up();
        minibatch.clean_up();
    }
    optimizer::outputs outs(loss_shots, acc_shots);
    return outs;
}

optimizer::outputs optimizer::Adam(DNN* dnn, Batch train_batch, Batch test_batch
    , size_t n_batch_size, size_t n_iteraions, size_t n_save_interval
    , double step_size, double beta1, double beta2, double epsilon, bool f_save) {
    std::vector<double>* loss_shots = new std::vector<double>;
    std::vector<double>* acc_shots = new std::vector<double>;
    if(f_save) {
        loss_shots->push_back(dnn->zero_oracle(train_batch));
        acc_shots->push_back(dnn->get_accuracy(test_batch));
    }
    std::vector<Tuple> m = dnn->get_zero_tuples();
    std::vector<Tuple> v = dnn->get_zero_tuples();
    for(size_t i = 0; i < n_iteraions; i ++) {
        Batch minibatch = random_batch_generator(train_batch, n_batch_size);
        std::vector<Tuple> grad = dnn->first_oracle(minibatch);

        double pow_beta1_i = pow(beta1, (double) (i + 1));
        double pow_beta2_i = pow(beta2, (double) (i + 1));
        double alpha = step_size * sqrt(1 - pow_beta2_i) / (1 - pow_beta1_i);
        for(size_t j = 0; j < dnn->get_n_layers() - 1; j ++) {
            m[j] *= (beta1 / (1 - pow_beta1_i));
            m[j](grad[j], ((1 - beta1) / (1 - pow_beta1_i)));

            v[j] *= (beta2 / (1 - pow_beta2_i));
            v[j]((grad[j] *= grad[j]), ((1 - beta2) / (1 - pow_beta2_i)));

            (m[j] *= -alpha) /= ((v[j].coeff_root()) += epsilon);
        }
        dnn->update_parameters(m);
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
        for(auto tuple : grad)
            tuple.clean_up();
        minibatch.clean_up();
    }
    // Clean up temp memory
    for(auto tuple : m)
        tuple.clean_up();
    for(auto tuple : v)
        tuple.clean_up();
    optimizer::outputs outs(loss_shots, acc_shots);
    return outs;
}

optimizer::outputs optimizer::SCR(DNN* dnn, Batch train_batch, Batch test_batch
    , size_t g_batch_size, size_t hv_batch_size, size_t n_iteraions, size_t sub_iterations
    , size_t n_save_interval, size_t petb_interval, double eta, double rho, double sigma
    , bool f_save) {
    std::vector<double>* loss_shots = new std::vector<double>;
    std::vector<double>* acc_shots = new std::vector<double>;
    if(f_save) {
        loss_shots->push_back(dnn->zero_oracle(train_batch));
        acc_shots->push_back(dnn->get_accuracy(test_batch));
    }
    size_t n_layers = dnn->get_n_layers();
    for(size_t i = 0; i < n_iteraions; i ++) {
        Batch g_batch = random_batch_generator(train_batch, g_batch_size);
        Batch hv_batch = random_batch_generator(train_batch, hv_batch_size);

        std::vector<Tuple> grad = dnn->first_oracle(g_batch);
        std::vector<Tuple> delta = dnn->get_zero_tuples();
        // Cubic Subproblem
        for(size_t j = 0; j < sub_iterations; j ++) {
            std::vector<Tuple> hv_delta = dnn->hessian_vector_oracle(hv_batch, delta);
            // Perturbe iterate every petb_interval steps
            // if(!(j % petb_interval)) {
            //     std::vector<Tuple> petb_tuples = dnn->get_perturb_tuples();
            //     for(size_t k = 0; k < n_layers - 1; k ++)
            //         delta[k](petb_tuples[k], sigma);
            //     // Clean up memory
            //     for(auto tuple : petb_tuples)
            //         tuple.clean_up();
            // }
            double delta_prefix = 0;
            for(size_t k = 0; k < n_layers - 1; k ++) {
                grad[k] += hv_delta[k];
                delta_prefix += delta[k].l2_norm_square();
            }
            delta_prefix = sqrt(delta_prefix) * rho / 2;
            for(size_t k = 0; k < n_layers - 1; k ++) {
                grad[k](delta[k], delta_prefix);
                delta[k](grad[k], -eta);
            }
            // Clean up memory
            for(auto tuple : hv_delta)
                tuple.clean_up();
        }
        dnn->update_parameters(delta);

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
        for(auto tuple : grad)
            tuple.clean_up();
        for(auto tuple : delta)
            tuple.clean_up();
        g_batch.clean_up();
        hv_batch.clean_up();
    }
    optimizer::outputs outs(loss_shots, acc_shots);
    return outs;
}
