#include <iostream>
#include "DNN.hpp"
#include "optimizer.hpp"
#include "activations.hpp"

size_t DIM, CLASS;
int main()
{
    NF = 3;
    DIM = 4;
    CLASS = 3;
    DNN dnn(1, new size_t[1]{1000}, 1, new double[1]{0.2}, 0.1);// sqrt(6 / (DIM + CLASS)));

    MatrixXr a(1,2);
    a.setRandom(1,2);
    std::cout << a << std::endl;

    MatrixXr X(3,4);
    X << 0,1,2,3,
         1,2,1.4,1.3,
         2,3,3,4;
    MatrixXr Y(3,1);
    Y << 0,1,0;
    int i;
    Y.col(0).maxCoeff(&i);
    std::cout << i << std::endl;
    // Batch batch(&X, &Y, 3);
    // std::cout << dnn.zero_oracle(batch) << std::endl;
    // std::vector<double>* res = optimizer::SGD(&dnn, &X, &Y, 1, 200, 1, 0.04, true);
    // for(double re : *res)
    //     std::cout << re << std::endl;
}
