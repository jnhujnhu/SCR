#include <iostream>
#include <string.h>
#include "global_def.hpp"
#include "DNN.hpp"
#include "optimizer.hpp"
#include "mex.h"

using namespace global_def;
size_t DIM, CLASS;
const size_t MAX_PARAM_STR_LEN = 15;
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    try {
        double *r_X = mxGetPr(prhs[0]);
        double *r_Y = mxGetPr(prhs[1]);
        double *r_XT = mxGetPr(prhs[2]);
        double *r_YT = mxGetPr(prhs[3]);
        size_t NF = mxGetN(prhs[0]);
        size_t NT = mxGetN(prhs[2]);
        DIM = mxGetM(prhs[0]);
        CLASS = (size_t) mxGetScalar(prhs[4]);
        size_t n_layers = (size_t) mxGetScalar(prhs[5]);
        double* i_stuc_layers = mxGetPr(prhs[6]);
        double lambda = mxGetScalar(prhs[8]);
        double batch_size = mxGetScalar(prhs[9]);
        size_t n_iteraions = (size_t) mxGetScalar(prhs[10]);
        size_t n_save_interval = (size_t) mxGetScalar(prhs[11]);
        double step_size = mxGetScalar(prhs[12]);
        bool f_save = false;
        if(nlhs >= 1)
            f_save = true;

        // Construct DNN Model
        size_t stuc_layers[n_layers];
        for(size_t i = 0; i < n_layers; i ++)
            stuc_layers[i] = (size_t) (int) i_stuc_layers[i];
        DNN dnn(n_layers, stuc_layers, 1, &lambda, 1.3 / sqrt((double)(DIM + CLASS)), I_GAUSSIAN);

        // Parse Data Matrices
        Eigen::Map<MatrixXr>* X = new Eigen::Map<MatrixXr>(r_X, NF, DIM);
        Eigen::Map<MatrixXr>* XT = new Eigen::Map<MatrixXr>(r_XT, NT, DIM);
        // Create 1-of-N Label Matrix
        MatrixXr* Y = new MatrixXr(NF, CLASS);
        MatrixXr* YT = new MatrixXr(NT, CLASS);
        Y->setZero(NF, CLASS);
        YT->setZero(NT, CLASS);
        for(size_t i = 0; i < NF; i ++)
            (*Y)(i, r_Y[i]) = 1;
        for(size_t i = 0; i < NT; i ++)
            (*YT)(i, r_YT[i]) = 1;
        Batch train_batch((MatrixXr *)X, Y, NF);
        Batch test_batch((MatrixXr *)XT, YT, NT);

        char* _algo = new char[MAX_PARAM_STR_LEN];
        mxGetString(prhs[7], _algo, MAX_PARAM_STR_LEN);
        double *stored_loss, *stored_acc;
        size_t len_stored_loss, len_stored_acc;
        if(strcmp(_algo, "SGD") == 0) {
            optimizer::outputs outs = optimizer::SGD(&dnn, train_batch
                , test_batch, batch_size, n_iteraions, n_save_interval, step_size
                , f_save);
            stored_loss = &(*outs._losses)[0];
            stored_acc = &(*outs._accuracies)[0];
            len_stored_loss = outs._losses->size();
            len_stored_acc = outs._accuracies->size();
        }
        else if(strcmp(_algo, "Adam") == 0) {
            double* adam_params = mxGetPr(prhs[13]);  // [Beta1, Beta2, Epsilon]
            optimizer::outputs outs = optimizer::Adam(&dnn, train_batch
                , test_batch, batch_size, n_iteraions, n_save_interval, step_size
                , adam_params[0], adam_params[1], adam_params[2], f_save);
            stored_loss = &(*outs._losses)[0];
            stored_acc = &(*outs._accuracies)[0];
            len_stored_loss = outs._losses->size();
            len_stored_acc = outs._accuracies->size();
        }
        else if(strcmp(_algo, "SCR") == 0) {
            // [hv_batch_size, sub_iterations, petb_interval, eta, rho, sigma]
            double* scr_params = mxGetPr(prhs[13]);
            size_t hv_batch_size = (size_t)(int) scr_params[0];
            size_t sub_iterations = (size_t)(int) scr_params[1];
            size_t petb_interval = (size_t)(int) scr_params[2];
            optimizer::outputs outs = optimizer::SCR(&dnn, train_batch
                , test_batch, batch_size, hv_batch_size, n_iteraions, sub_iterations
                , n_save_interval, petb_interval, scr_params[3], scr_params[4]
                , scr_params[5], f_save);
            stored_loss = &(*outs._losses)[0];
            stored_acc = &(*outs._accuracies)[0];
            len_stored_loss = outs._losses->size();
            len_stored_acc = outs._accuracies->size();
        }
        else mexErrMsgTxt("400 Unrecognized algorithm.");
        delete[] _algo;

        if(f_save) {
            plhs[0] = mxCreateDoubleMatrix(len_stored_loss, 1, mxREAL);
            plhs[1] = mxCreateDoubleMatrix(len_stored_acc, 1, mxREAL);
        	double* res_stored_loss = mxGetPr(plhs[0]);
            double* res_stored_acc = mxGetPr(plhs[1]);
            for(size_t i = 0; i < len_stored_loss; i ++)
                res_stored_loss[i] = stored_loss[i];
            for(size_t i = 0; i < len_stored_acc; i ++)
                res_stored_acc[i] = stored_acc[i];
        }
        delete[] stored_loss;
        delete[] stored_acc;
    } catch(std::string c) {
        std::cerr << c << std::endl;
        //exit(EXIT_FAILURE);
    }
}
