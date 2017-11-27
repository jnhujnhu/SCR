#include <iostream>
#include "DNN.hpp"
#include "optimizer.hpp"
#include "activations.hpp"

size_t DIM, CLASS;

int main()
{
    DIM = 4;
    CLASS = 3;
    DNN dnn(1, new size_t[1]{10}, 1, new double[1]{0.2}, 0.2);// sqrt(6 / (DIM + CLASS)));

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
    std::vector<Tuple> res = dnn.hessian_vector_oracle(batch, dnn.get_ones_tuples());
    for(auto tuple : res) {
        tuple.print_all();
    }
    // optimizer::SCR(&dnn, batch, batch, 3, 3, 200, 10, 1, 1, 0.05, 0, 1.1, true);
}
