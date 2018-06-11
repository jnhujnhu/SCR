#include <iostream>
#include "DNN.hpp"
#include "optimizer.hpp"
#include "activations.hpp"

size_t DIM, CLASS;
// Symmetric PD
MatrixXr H(4,4);

MatrixXr fake_hv(MatrixXr V) {
    return H * H * V;
}

MatrixXr HV(MatrixXr V) {
    return H * V;
}

double gauss_unary(double dummy) {
    std::random_device rd;
    std::mt19937 generator(rd());
    std::normal_distribution<double> distribution(0, 1);
    return distribution(generator);
}

size_t oracle_counter = 0;
// Validate approximation of inverse root of squared hessian vector product
MatrixXr C(size_t depth, MatrixXr g) {
    if(depth == 0) {
        return g;
    }
    oracle_counter ++;
    return 1.5 * C(depth - 1, g - 1.0 / 3.0 * C(depth - 1, C(depth - 1, fake_hv(g))));
}

// Lanczos Iteration Test
MatrixXr Lanczos(MatrixXr (*MV)(MatrixXr), size_t iter_count) {
    // Initialize
    MatrixXr w(DIM, 1);
    w = w.unaryExpr(std::ptr_fun(gauss_unary));
    w.normalize();
    std::cout << w << std::endl;
    return w;
}

int main()
{
//Lanczos Iteration
    // DIM = 10;
    // H = MatrixXr::Random(DIM,DIM);
    // MatrixXr V = MatrixXr::Random(DIM, 1);
    // Lanczos(HV, 10);

//Inverse root of squared hessian vector product
    // H << 5,   1,  2,   0.33,
    //      1,   8,  1.4, 1.3,
    //      2,   1.4,3,   1.34,
    //      0.33,1.3,1.34,2.23;
    // H = H / 6;
    // // Check PDness
    // //  Eigen::SelfAdjointEigenSolver<MatrixXr> eigensolver(H);
    // //  if (eigensolver.info() != Eigen::Success) abort();
    // //  std::cout << eigensolver.eigenvalues() << std::endl;
    // MatrixXr g(4,1);
    // g << 1,
    //      8,
    //      1,
    //      2;
    // MatrixXr res = C(6, g);
    // std::cout << "Approximate: " << std::endl << res << std::endl
    //     << "Exact: " << std::endl << H.inverse() * g << std::endl
    //     << oracle_counter << std::endl;


    DIM = 4;
    CLASS = 3;
    DNN dnn(2, new size_t[2]{10, 5}, 1, new double[1]{0.2}, 0.01, I_GAUSSIAN);// sqrt(6 / (DIM + CLASS)));
    std::vector<Tuple> petb = dnn.get_perturb_tuples();
    for(size_t j = 0; j < dnn.get_n_layers() - 1; j ++)
        petb[j].print_all();

    // // std::cout << a << std::endl;
    //
    // MatrixXr X(3,4);
    // X << 0,1,2,3,
    //      1,2,1.4,1.3,
    //      2,3,3,4;
    // MatrixXr Y(3,3);
    // Y << 0,1,0,
    //      1,0,0,
    //      0,0,1;
    // Batch batch(&X, &Y, 3);
// Hessian Vector Check
    // std::vector<Tuple> grad = dnn.first_oracle(batch);
    // std::vector<Tuple> petb_grad = dnn.perturbed_batch_first_oracle(batch, 0.000001);
    // for(size_t j = 0; j < dnn.get_n_layers() - 1; j ++) {
    //     grad[j].print_all();
    // }
    // std::cout << "#######################################" << std::endl;
    // for(size_t j = 0; j < dnn.get_n_layers() - 1; j ++) {
    //    petb_grad[j].print_all();
    // }
    // std::vector<Tuple> delta2 = dnn.get_perturb_tuples();
    // std::vector<Tuple> delta = dnn.get_ones_tuples();
    // for(size_t j = 0; j < dnn.get_n_layers() - 1; j ++) {
    //     (delta2[j] *= 3.141592653589) += delta[j];
    //     delta[j] = delta2[j];
    // }
    // std::vector<Tuple> res = dnn.hessian_vector_oracle(batch, delta);
    // std::vector<Tuple> res2 = dnn.hessian_vector_approxiamate_oracle(batch, grad, delta);
    // double a = 0, a2 = 0;
    // for(size_t j = 0; j < dnn.get_n_layers() - 1; j ++) {
    //     delta[j] *= res[j];
    //     delta2[j] *= res2[j];
    //     a += delta[j].sum();
    //     a2 += delta2[j].sum();
    // }
    // std::cout << a << std::endl << a2 << std::endl;
// Check Memory Leak
    // optimizer::SCR(&dnn, batch, batch, 3, 3, 200000, 10, 1, 0, 0.05, 0.1, 1.1, true);
    // optimizer::Adam(&dnn, batch, batch, 3, 200000, 10, 0.01, 0.9, 0.999, 1e-08, true);
    // optimizer::AdaGrad(&dnn, batch, batch, 3, 200000, 10, 0.5, 1e-08, true);
    // optimizer::SGD(&dnn, batch, batch, 3, 200000, 10, 0.5, 0, true);
}
