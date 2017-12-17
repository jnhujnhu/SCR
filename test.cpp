#include <iostream>
#include "DNN.hpp"
#include "optimizer.hpp"
#include "activations.hpp"

size_t DIM, CLASS;

int main()
{
    DIM = 4;
    CLASS = 3;
    DNN dnn(2, new size_t[2]{10, 5}, 1, new double[1]{0.2}, 0.01, I_GAUSSIAN);// sqrt(6 / (DIM + CLASS)));

    // std::cout << a << std::endl;

    MatrixXr X(3,4);
    X << 0,1,2,3,
         1,2,1.4,1.3,
         2,3,3,4;
    MatrixXr Y(3,3);
    Y << 0,1,0,
         1,0,0,
         0,0,1;
    Batch batch(&X, &Y, 3);
    // std::vector<Tuple> grad = dnn.first_oracle(batch);
    // std::vector<Tuple> delta2 = dnn.get_perturb_tuples();
    // std::vector<Tuple> delta = dnn.get_ones_tuples();
    // for(size_t j = 0; j < dnn.get_n_layers() - 1; j ++) {
    //     (delta2[j] *= 3) += delta[j];
    //     delta[j] = delta2[j];
    // }
    // // Hessian Vector Check
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
    optimizer::SGD(&dnn, batch, batch, 3, 200000, 10, 0.5, 0, true);
}
