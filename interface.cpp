#include <iostream>
#include <string.h>
#include "global_def.hpp"
#include "DNN.hpp"
#include "optimizer.hpp"
#include "mex.h"

using namespace global_def;
size_t DIM, NF, CLASS;
const size_t MAX_PARAM_STR_LEN = 15;
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    try {
        double *r_X = mxGetPr(prhs[0]);
        double *r_Y = mxGetPr(prhs[1]);
        DIM = mxGetM(prhs[0]);
        NF = mxGetN(prhs[0]);
        CLASS = (size_t) mxGetScalar(prhs[2]);
        size_t n_layers = (size_t) mxGetScalar(prhs[3]);
        double* i_stuc_layers = mxGetPr(prhs[4]);
        double lambda = mxGetScalar(prhs[6]);
        double batch_size = mxGetScalar(prhs[7]);
        size_t n_iteraions = (size_t) mxGetScalar(prhs[8]);
        size_t n_save_interval = (size_t) mxGetScalar(prhs[9]);
        double step_size = mxGetScalar(prhs[10]);
        bool f_save = false;
        if(nlhs == 1)
            f_save = true;

        // Construct DNN Model
        size_t stuc_layers[n_layers];
        for(size_t i = 0; i < n_layers; i ++)
            stuc_layers[i] = (size_t) (int) i_stuc_layers[i];
        DNN dnn(n_layers, stuc_layers, 1, &lambda, sqrt(6 / (DIM + CLASS)));

        // Parse Data Matrices
        Eigen::Map<MatrixXr>* X = new Eigen::Map<MatrixXr>(r_X, NF, DIM);
        // Create 1-of-N Label Matrix
        MatrixXr* Y = new MatrixXr(NF, CLASS);
        Y->setZero(NF, CLASS);
        for(size_t i = 0; i < NF; i ++)
            (*Y)(i, r_Y[i]) = 1;

        char* _algo = new char[MAX_PARAM_STR_LEN];
        mxGetString(prhs[5], _algo, MAX_PARAM_STR_LEN);
        double* stored;
        size_t len_stored;
        if(strcmp(_algo, "SGD") == 0) {
            std::vector<double>* vec_stored = optimizer::SGD(&dnn, (MatrixXr*) X, Y, batch_size, n_iteraions,
                n_save_interval, step_size, f_save);
            stored = &(*vec_stored)[0];
            len_stored = vec_stored->size();
        }
        else if(strcmp(_algo, "ADAM") == 0) {
        }
        else if(strcmp(_algo, "SCR") == 0) {
        }
        else mexErrMsgTxt("400 Unrecognized algorithm.");
        delete[] _algo;

        if(f_save) {
            plhs[0] = mxCreateDoubleMatrix(len_stored, 1, mxREAL);
        	double* res_stored = mxGetPr(plhs[0]);
            for(size_t i = 0; i < len_stored; i ++)
                res_stored[i] = stored[i];
        }
        delete[] stored;
    } catch(std::string c) {
        std::cerr << c << std::endl;
        //exit(EXIT_FAILURE);
    }
}
