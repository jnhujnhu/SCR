#include "optimizer.hpp"
#include <random>

Batch optimizer::random_batch_generator(Batch full_batch, size_t batch_size
    , bool is_autoencoder) {
    // Random Generator
    std::random_device rd;
    std::mt19937 generator(rd());
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

void optimizer::deliberate_perturbe(DNN *dnn, std::vector<Tuple> target, double std_dev) {
    std::vector<Tuple> petb_tuples = dnn->get_perturb_tuples();
    for(size_t k = 0; k < dnn->get_n_layers() - 1; k ++)
        target[k](petb_tuples[k], std_dev);
    // Clean up temp memory
    for(auto tuple : petb_tuples)
        tuple.clean_up();
}

optimizer::outputs optimizer::SGD(DNN* dnn, Batch train_batch, Batch test_batch
    , size_t n_batch_size, size_t n_iteraions, size_t n_save_interval, double step_size
    , double decay, bool using_saddle_free_gradient, bool using_petb_iterate
    , bool using_petb_batch, double petb_radius, bool f_save) {
    std::vector<double>* loss_shots = new std::vector<double>;
    std::vector<double>* acc_shots = new std::vector<double>;
    if(f_save) {
        loss_shots->push_back(dnn->zero_oracle(train_batch));
        if(!dnn->isAutoencoder())
            acc_shots->push_back(dnn->get_accuracy(test_batch));
    }

    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_real_distribution<> distribution(0.0, petb_radius);
    std::vector<Tuple> snapshot;
    // Create snapshot point
    if(using_saddle_free_gradient)
        snapshot = dnn->get_param_tuples_copy();
    for(size_t i = 0; i < n_iteraions; i ++) {
        Batch minibatch = random_batch_generator(train_batch, n_batch_size
            , dnn->isAutoencoder());
        std::vector<Tuple> grad;
        if(using_saddle_free_gradient && !(i % 15)) {
            std::vector<Tuple> params = dnn->get_param_tuples_copy();
            grad = dnn->first_oracle(minibatch, &snapshot);
            for(size_t j = 0; j < dnn->get_n_layers() - 1; j ++)
                params[j] -= snapshot[j];
            // Compute H^2(v)
            std::vector<Tuple> h_p = dnn->hessian_vector_oracle(minibatch, params, &snapshot);
            std::vector<Tuple> hh_p = dnn->hessian_vector_oracle(minibatch, h_p, &snapshot);
            for(size_t j = 0; j < dnn->get_n_layers() - 1; j ++) {
                grad[j] += hh_p[j];
                params[j].clean_up();
                h_p[j].clean_up();
                hh_p[j].clean_up();
            }
            // Update snapshot point
            snapshot = dnn->get_param_tuples_copy();
        }
        // Methods of Adding Perturbation
        else if(using_petb_batch)
            grad = dnn->perturbed_batch_first_oracle(minibatch, petb_radius);
        else if(using_petb_iterate) {
            grad = dnn->first_oracle(minibatch);
            deliberate_perturbe(dnn, grad, distribution(generator));
        }
        else
            grad = dnn->first_oracle(minibatch);
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
            // For Hessian_Vector_Approxiamate
            // std::vector<Tuple> hv_grad = dnn->first_oracle(hv_batch);
            // Perturbe method mentioned in Paper
            if(!petb_interval)
                deliberate_perturbe(dnn, grad, sigma);
            for(size_t j = 0; j < sub_iterations; j ++) {
                // Hessian_Vector_Approxiamate
                // std::vector<Tuple> hv_delta
                //     = dnn->hessian_vector_approxiamate_oracle(hv_batch, hv_grad, delta);
                // Hessian_Vector_Exact
                std::vector<Tuple> hv_delta = dnn->hessian_vector_oracle(hv_batch, delta);
                std::vector<Tuple> grad_sub = dnn->get_zero_tuples();
                // Perturbe iterate every petb_interval steps
                if(petb_interval && !(j % petb_interval))
                    deliberate_perturbe(dnn, delta, sigma);
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
                for(Tuple gs_tuple : grad_sub)
                    gs_tuple.clean_up();
            }
            // For Hessian_Vector_Approxiamate
            // for(auto tuple : hv_grad)
            //     tuple.clean_up();
        }
        dnn->update_parameters(delta);
        if(f_save && !(i % n_save_interval)) {
            if(!standard_trace(dnn, i, train_batch, test_batch, loss_shots
                , acc_shots)) {
                std::cerr << "NaN Occurred." << std::endl;
                return optimizer::outputs(loss_shots, acc_shots);
            }
        }
        // Clean up temp memory
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



// optimizer::outputs optimizer::TSCSG(DNN* dnn, Batch train_batch, Batch test_batch
//     , size_t batch_size, size_t minibatch_size, size_t n_iteraions, size_t n_save_interval
//     , double step_size, bool f_save) {
//     std::vector<double>* loss_shots = new std::vector<double>;
//     std::vector<double>* acc_shots = new std::vector<double>;
//     if(f_save) {
//         loss_shots->push_back(dnn->zero_oracle(train_batch));
//         if(!dnn->isAutoencoder())
//             acc_shots->push_back(dnn->get_accuracy(test_batch));
//     }
//     size_t n_layers = dnn->get_n_layers();
//     size_t sub_iterations = 2 * train_batch._n;
//     for(size_t i = 0; i < n_iteraions; i ++) {
//         Batch batch = random_batch_generator(train_batch, batch_size
//             , dnn->isAutoencoder());
//         std::vector<Tuple> batch_grad = dnn->first_oracle(batch);
//         for(size_t j = 0; j < sub_iterations; j ++) {
//             Batch minibatch = random_batch_generator(train_batch, minibatch_size,
//                 dnn->isAutoencoder());
//             std::vector<Tuple> grad = dnn->first_oracle(batch);
//             std::vector<Tuple> delta = dnn->get_zero_tuples();
//             for(size_t k = 0; k < n_layers - 1; k ++) {
//                 delta[k] = grad[k];
//                 delta[k] -= ;
//             }
//             std::vector<Tuple> Hg = inverse_square_root_finder(5, grad, dnn, hv_batch);
//             for(size_t k = 0; k < n_layers - 1; k ++)
//                 delta[k](Hg[k], - 1.5);
//         }
//         dnn->update_parameters(delta);
//         if(f_save && !(i % n_save_interval)) {
//             if(!standard_trace(dnn, i, train_batch, test_batch, loss_shots
//                 , acc_shots)) {
//                 std::cerr << "NaN Occurred." << std::endl;
//                 return optimizer::outputs(loss_shots, acc_shots);
//             }
//         }
//         // Clean up temp memory
//         for(size_t k = 0; k < n_layers - 1; k ++) {
//             grad[k].clean_up();
//             delta[k].clean_up();
//             Hg[k].clean_up();
//         }
//         batch.clean_up();
//         hv_batch.clean_up();
//     }
//     optimizer::outputs outs(loss_shots, acc_shots);
//     return outs;
// }

std::vector<Tuple> squared_hessian_vector_oracle(DNN* dnn, Batch hv_batch
    , std::vector<Tuple> V) {
    std::vector<Tuple> Hv = dnn->hessian_vector_oracle(hv_batch, V);
    std::vector<Tuple> HHv = dnn->hessian_vector_oracle(hv_batch, Hv);
    for(Tuple tuple : Hv) tuple.clean_up();
    for(Tuple tuple : HHv) {
        tuple *= 4.0 / 9.0;
    }
    return HHv;
}

std::vector<Tuple> inverse_square_root_finder(size_t depth, std::vector<Tuple> g
    , DNN* dnn, Batch hv_batch) {
    if(depth == 0)
        return g;
    std::vector<Tuple> hvo = squared_hessian_vector_oracle(dnn, hv_batch, g);
    std::vector<Tuple> temp = inverse_square_root_finder(depth - 1
                                , inverse_square_root_finder(depth - 1
                                    , hvo
                                , dnn, hv_batch)
                            , dnn, hv_batch);
    for(Tuple tuple : hvo)
        tuple.clean_up();
    for(size_t i = 0; i < dnn->get_n_layers() - 1; i ++) {
        temp[i] *= (1.0 / 3.0);
        Tuple temp_2;
        temp_2 = g[i];
        temp_2 -= temp[i];
        temp[i] = temp_2;
        temp_2.clean_up();
    }
    std::vector<Tuple> res = inverse_square_root_finder(depth - 1, temp, dnn, hv_batch);
    for(Tuple tuple : temp)
        tuple.clean_up();
    for(Tuple tuple : res)
        tuple *= 1.5;
    return res;
}

// Under Construction
// Stochastic Saddle Free Newton (TODO)
optimizer::outputs optimizer::SSFN(DNN* dnn, Batch train_batch, Batch test_batch
    , size_t g_batch_size, size_t hv_batch_size, size_t n_iteraions
    , size_t n_save_interval, double L, double epsilon, bool f_save) {
    std::vector<double>* loss_shots = new std::vector<double>;
    std::vector<double>* acc_shots = new std::vector<double>;
    if(f_save) {
        loss_shots->push_back(dnn->zero_oracle(train_batch));
        if(!dnn->isAutoencoder())
            acc_shots->push_back(dnn->get_accuracy(test_batch));
    }
    size_t n_layers = dnn->get_n_layers();
    for(size_t i = 0; i < n_iteraions; i ++) {
        Batch g_batch = random_batch_generator(train_batch, g_batch_size
            , dnn->isAutoencoder());
        Batch hv_batch = random_batch_generator(train_batch, hv_batch_size
            , dnn->isAutoencoder());

        std::vector<Tuple> grad = dnn->first_oracle(g_batch);
        std::vector<Tuple> delta = dnn->get_zero_tuples();

        std::vector<Tuple> Hg = inverse_square_root_finder(5, grad, dnn, hv_batch);
        for(size_t k = 0; k < n_layers - 1; k ++)
            delta[k](Hg[k], - 1.5);

        dnn->update_parameters(delta);
        if(f_save && !(i % n_save_interval)) {
            if(!standard_trace(dnn, i, train_batch, test_batch, loss_shots
                , acc_shots)) {
                std::cerr << "NaN Occurred." << std::endl;
                return optimizer::outputs(loss_shots, acc_shots);
            }
        }
        // Clean up temp memory
        for(size_t k = 0; k < n_layers - 1; k ++) {
            grad[k].clean_up();
            delta[k].clean_up();
            Hg[k].clean_up();
        }
        g_batch.clean_up();
        hv_batch.clean_up();
    }
    optimizer::outputs outs(loss_shots, acc_shots);
    return outs;
}
