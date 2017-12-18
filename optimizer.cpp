#include "optimizer.hpp"
#include <random>

Batch optimizer::random_batch_generator(Batch full_batch, size_t batch_size
    , bool is_autoencoder) {
    // Random Generator
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::vector<int> indexes;
    MatrixXr* _X = new MatrixXr(batch_size, DIM);
    MatrixXr* _Y;
    if(is_autoencoder)
        _Y = new MatrixXr(batch_size, DIM);
    else
        _Y = new MatrixXr(batch_size, CLASS);
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
}

bool optimizer::standard_trace(DNN* dnn, size_t i, Batch train_batch, Batch test_batch
    , std::vector<double>* loss_shots, std::vector<double>* acc_shots) {
    double loss = dnn->zero_oracle(train_batch);
    double acc;
    loss_shots->push_back(loss);
    if(!dnn->isAutoencoder()) {
        acc = dnn->get_accuracy(test_batch);
        acc_shots->push_back(acc);
    }
    std::cout.precision(13);
    std::cout << "Iteration " << i << " with loss = " << loss;
    if(!dnn->isAutoencoder())
        std::cout << " acc = " << acc << std::endl;
    else
        std::cout << std::endl;
    if(std::isnan(loss))
        return false;
    else
        return true;
}

optimizer::outputs optimizer::SGD(DNN* dnn, Batch train_batch, Batch test_batch
    , size_t n_batch_size, size_t n_iteraions, size_t n_save_interval, double step_size
    , double decay, bool f_save) {
    std::vector<double>* loss_shots = new std::vector<double>;
    std::vector<double>* acc_shots = new std::vector<double>;
    if(f_save) {
        loss_shots->push_back(dnn->zero_oracle(train_batch));
        if(!dnn->isAutoencoder())
            acc_shots->push_back(dnn->get_accuracy(test_batch));
    }
    for(size_t i = 0; i < n_iteraions; i ++) {
        Batch minibatch = random_batch_generator(train_batch, n_batch_size
            , dnn->isAutoencoder());
        std::vector<Tuple> grad = dnn->first_oracle(minibatch);
        // Adopt Decay Scheme in Keras SGD.
        step_size *= 1.0 / (1.0 + decay * (i + 1));
        for(auto tuple : grad)
            tuple *= -step_size;
        dnn->update_parameters(grad);
        if(f_save && !(i % n_save_interval)) {
            if(!standard_trace(dnn, i, train_batch, test_batch, loss_shots
                , acc_shots)) {
                std::cerr << "NaN Occurred." << std::endl;
                return optimizer::outputs(loss_shots, acc_shots);
            }
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
        if(!dnn->isAutoencoder())
            acc_shots->push_back(dnn->get_accuracy(test_batch));
    }
    std::vector<Tuple> m = dnn->get_zero_tuples();
    std::vector<Tuple> v = dnn->get_zero_tuples();
    // Temp Variables
    std::vector<Tuple> incr = dnn->get_zero_tuples();
    double pow_beta1_i = beta1;
    double pow_beta2_i = beta2;
    for(size_t i = 0; i < n_iteraions; i ++) {
        Batch minibatch = random_batch_generator(train_batch, n_batch_size
            , dnn->isAutoencoder());
        std::vector<Tuple> grad = dnn->first_oracle(minibatch);
        double alpha = step_size * sqrt(1.0 - pow_beta2_i) / (1.0 - pow_beta1_i);
        for(size_t j = 0; j < dnn->get_n_layers() - 1; j ++) {
            m[j] *= beta1;
            m[j](grad[j], 1.0 - beta1);

            v[j] *= beta2;
            grad[j] *= grad[j];
            v[j](grad[j], 1.0 - beta2);

            Tuple temp;
            temp = v[j];
            temp.coeff_root();
            temp += epsilon;

            incr[j] = m[j];
            incr[j] /= temp;
            incr[j] *= -alpha;

            temp.clean_up();
        }
        dnn->update_parameters(incr);
        pow_beta1_i *= beta1;
        pow_beta2_i *= beta2;
        if(f_save && !(i % n_save_interval)) {
            if(!standard_trace(dnn, i, train_batch, test_batch, loss_shots
                , acc_shots)) {
                std::cerr << "NaN Occurred." << std::endl;
                return optimizer::outputs(loss_shots, acc_shots);
            }
        }
        // Clean up temp memory
        for(auto tuple : grad)
            tuple.clean_up();
        minibatch.clean_up();
    }
    // Clean up temp memory
    for(size_t j = 0; j < dnn->get_n_layers() - 1; j ++) {
        m[j].clean_up();
        v[j].clean_up();
        incr[j].clean_up();
    }
    optimizer::outputs outs(loss_shots, acc_shots);
    return outs;
}

optimizer::outputs optimizer::AdaGrad(DNN* dnn, Batch train_batch, Batch test_batch
    , size_t n_batch_size, size_t n_iteraions, size_t n_save_interval
    , double step_size, double epsilon, bool f_save) {
    std::vector<double>* loss_shots = new std::vector<double>;
    std::vector<double>* acc_shots = new std::vector<double>;
    if(f_save) {
        loss_shots->push_back(dnn->zero_oracle(train_batch));
        if(!dnn->isAutoencoder())
            acc_shots->push_back(dnn->get_accuracy(test_batch));
    }
    std::vector<Tuple> g2_accumulator = dnn->get_zero_tuples();
    // Temp Variables
    std::vector<Tuple> incr = dnn->get_zero_tuples();
    for(size_t i = 0; i < n_iteraions; i ++) {
        Batch minibatch = random_batch_generator(train_batch, n_batch_size
            , dnn->isAutoencoder());
        std::vector<Tuple> grad = dnn->first_oracle(minibatch);
        for(size_t j = 0; j < dnn->get_n_layers() - 1; j ++) {
            Tuple temp;
            temp = grad[j];
            temp *= temp;
            g2_accumulator[j] += temp;
            // First Step using SGD
            if(i != 0) {
                temp = g2_accumulator[j];
                temp += epsilon;
                temp.coeff_root();
                temp.reciprocal();
            }

            incr[j] = grad[j];
            if(i != 0)
                incr[j] *= temp;
            incr[j] *= -step_size;
            temp.clean_up();
        }
        dnn->update_parameters(incr);

        if(f_save && !(i % n_save_interval)) {
            if(!standard_trace(dnn, i, train_batch, test_batch, loss_shots
                , acc_shots)) {
                std::cerr << "NaN Occurred." << std::endl;
                return optimizer::outputs(loss_shots, acc_shots);
            }
        }
        // Clean up temp memory
        for(auto tuple : grad)
            tuple.clean_up();
        minibatch.clean_up();
    }
    // Clean up temp memory
    for(Tuple tuple : incr) {
        tuple.clean_up();
    }
    optimizer::outputs outs(loss_shots, acc_shots);
    return outs;
}

optimizer::outputs optimizer::SCR(DNN* dnn, Batch train_batch, Batch test_batch
    , size_t g_batch_size, size_t hv_batch_size, size_t n_iteraions, size_t sub_iterations
    , size_t n_save_interval, size_t petb_interval, double L, double rho, double sigma
    , bool f_save) {
    std::vector<double>* loss_shots = new std::vector<double>;
    std::vector<double>* acc_shots = new std::vector<double>;
    if(f_save) {
        loss_shots->push_back(dnn->zero_oracle(train_batch));
        if(!dnn->isAutoencoder())
            acc_shots->push_back(dnn->get_accuracy(test_batch));
    }
    size_t n_layers = dnn->get_n_layers();
    size_t cauchy_step_counter = 0;
    for(size_t i = 0; i < n_iteraions; i ++) {
        Batch g_batch = random_batch_generator(train_batch, g_batch_size
            , dnn->isAutoencoder());
        Batch hv_batch = random_batch_generator(train_batch, hv_batch_size
            , dnn->isAutoencoder());

        std::vector<Tuple> grad = dnn->first_oracle(g_batch);
        std::vector<Tuple> delta = dnn->get_zero_tuples();
        // Cubic Subsolver
        double grad_norm = 0;
        for(Tuple g_tuple : grad)
            grad_norm += g_tuple.l2_norm_square();
        // Perform One Cauchy Step in Large Gradient Norm Case
        if(sqrt(grad_norm) >= L * L / rho) {
            cauchy_step_counter ++;
            std::vector<Tuple> hv_grad = dnn->hessian_vector_oracle(hv_batch, grad);
            double grad_quad = 0;
            for(size_t j = 0; j < n_layers - 1; j ++) {
                hv_grad[j] *= grad[j];
                grad_quad += hv_grad[j].sum();
            }
            double cauchy_radius = grad_quad / (rho * grad_norm);
            cauchy_radius = -cauchy_radius + sqrt(cauchy_radius * cauchy_radius
                + 2 * sqrt(grad_norm) / rho);
            for(size_t j = 0; j < n_layers - 1; j ++)
                delta[j](grad[j], -cauchy_radius / sqrt(grad_norm));
            // Clean up memory
            for(Tuple hv_tuple : hv_grad)
                hv_tuple.clean_up();
        }
        // Else Tackle subproblem using Gradient Descent
        else {
            double eta = 1.0 / (20 * L);
            std::vector<Tuple> grad_sub = dnn->get_zero_tuples();
            // // For Hessian_Vector_Approxiamate
            // std::vector<Tuple> hv_grad = dnn->first_oracle(hv_batch);
            // Perturbe method mentioned in Paper
            if(!petb_interval) {
                std::vector<Tuple> petb_tuples = dnn->get_perturb_tuples();
                for(size_t k = 0; k < n_layers - 1; k ++)
                    grad[k](petb_tuples[k], sigma);
                // Clean up memory
                for(auto tuple : petb_tuples)
                    tuple.clean_up();
            }
            for(size_t j = 0; j < sub_iterations; j ++) {
                // Hessian_Vector_Approxiamate
                // std::vector<Tuple> hv_delta
                //     = dnn->hessian_vector_approxiamate_oracle(hv_batch, hv_grad, delta);
                // Hessian_Vector_Exact
                std::vector<Tuple> hv_delta = dnn->hessian_vector_oracle(hv_batch, delta);
                // Perturbe iterate every petb_interval steps
                if(petb_interval && !(j % petb_interval)) {
                    std::vector<Tuple> petb_tuples = dnn->get_perturb_tuples();
                    for(size_t k = 0; k < n_layers - 1; k ++)
                        delta[k](petb_tuples[k], sigma);
                    // Clean up memory
                    for(auto tuple : petb_tuples)
                        tuple.clean_up();
                }
                double delta_prefix = 0;
                for(size_t k = 0; k < n_layers - 1; k ++) {
                    grad_sub[k] = grad[k];
                    grad_sub[k] += hv_delta[k];
                    delta_prefix += delta[k].l2_norm_square();
                }
                delta_prefix = sqrt(delta_prefix) * rho / 2;
                for(size_t k = 0; k < n_layers - 1; k ++) {
                    grad_sub[k](delta[k], delta_prefix);
                    delta[k](grad_sub[k], -eta);
                }
                // Clean up memory
                for(Tuple hv_tuple : hv_delta)
                    hv_tuple.clean_up();
            }
            for(Tuple gs_tuple : grad_sub) {
                gs_tuple.clean_up();
            }
            // // For Hessian_Vector_Approxiamate
            // for(auto tuple : hv_grad)
            //     tuple.clean_up();
            // Clean up temp memory
        }
        dnn->update_parameters(delta);
        if(f_save && !(i % n_save_interval)) {
            if(!standard_trace(dnn, i, train_batch, test_batch, loss_shots
                , acc_shots)) {
                std::cerr << "NaN Occurred." << std::endl;
                return optimizer::outputs(loss_shots, acc_shots);
            }
        }
        for(size_t k = 0; k < n_layers - 1; k ++) {
            grad[k].clean_up();
            delta[k].clean_up();
        }
        g_batch.clean_up();
        hv_batch.clean_up();
    }
    std::cout << "SCR performed " << cauchy_step_counter << " steps Cauchy Steps."
        << std::endl;
    optimizer::outputs outs(loss_shots, acc_shots);
    return outs;
}
