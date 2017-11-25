#include <iostream>
#include "DNN.hpp"
#include "optimizer.hpp"

size_t DIM, CLASS, NF;
int main()
{
    NF = 3;
    DIM = 3;
    CLASS = 2;
    DNN dnn(2, new size_t[2]{50,60}, 1, new double[1]{0.2}, sqrt(6 / (DIM + CLASS)));
    MatrixXr X(3,3);
    X << 1,2,3,
         4,5,6,
         7,8,9;
    MatrixXr Y(3,2);
    Y << 1,0,
         0,1,
         1,0;
    std::vector<double>* res = optimizer::SGD(&dnn, &X, &Y, 1, 200, 1, 0.04, true);
    for(double re : *res)
        std::cout << re << std::endl;
}
