#ifndef GLOBAL_HPP
#define GLOBAL_HPP
#include <Eigen/Dense>
#include <iostream>

extern size_t DIM;
extern size_t CLASS;

namespace global_def {
    typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXr;
    typedef Eigen::Matrix<double, Eigen::Dynamic, 1> VectorXr;

    // Regularizers
    const int NONE = -1;
    const int L1 = 0;
    const int L2 = 1;
    const int ELASTIC_NET = 2;

    // Initializer
    const size_t I_GAUSSIAN = 1;
    const size_t I_UNIFORM = 2;
    const size_t I_ZERO = 3;

    class Tuple {
    public:
        Tuple(MatrixXr* weights, VectorXr* biases): _w(weights), _b(biases) {}
        void clean_up();
        void print_all();
        Tuple& operator +=(const Tuple& rhs);
        Tuple& operator -=(const Tuple& rhs);
        Tuple& operator *=(const Tuple& rhs);
        Tuple& operator /=(const Tuple& rhs);
        Tuple& operator +=(const double rhs);
        Tuple& operator *=(const double rhs);
        void operator ()(const Tuple& rhs, const double scalar);

        double l2_norm_square();
        Tuple& coeff_root();

        MatrixXr* _w;
        VectorXr* _b;
    };

    class Batch {
    public:
        Batch(MatrixXr* X, MatrixXr* Y, size_t N): _X(X), _Y(Y), _n(N) {}
        void clean_up();
        MatrixXr* _X;
        MatrixXr* _Y;
        size_t _n;
    };
}

#endif
