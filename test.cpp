#include <iostream>
#include "DNN.hpp"
#include "optimizer.hpp"
#include "activations.hpp"

size_t DIM, CLASS;
int main()
{
    DIM = 4;
    CLASS = 3;
    DNN dnn(1, new size_t[1]{10}, 1, new double[1]{0.2}, 0.1);// sqrt(6 / (DIM + CLASS)));

    // std::cout << a << std::endl;

    // MatrixXr X(3,4);
    // X << 0,1,2,3,
    //      1,2,1.4,1.3,
    //      2,3,3,4;
    // MatrixXr Y(3,3);
    // Y << 0,1,0,
    //      1,0,0,
    //      0,0,1;

    MatrixXr X(3,4);
    X << 0,1,2,3,
         1,2,1.4,1.3,
         2,3,3,4;
    VectorXr Y(3,1);
    Y << 0,1,0;

    MatrixXr Z(3,4);
    Z << 0,1,2,3,
         1,2,1.4,1.3,
         2,3,3,4;
    VectorXr K(3,1);
    K << 0,1,0;
    Tuple a(&Z, &K), b(&X, &Y);
    b *= b;

    b.print_all();
    //b.print_all();
    // Batch batch(&X, &Y, 3);
    // optimizer::Adam(&dnn, batch, batch , 1, 200, 1, 0.04, 0.9, 0.999, 1e-8, true);
}
