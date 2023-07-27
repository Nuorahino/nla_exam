#ifndef TEST_CAS_HH
#define TEST_CAS_HH

#include <eigen3/Eigen/Dense>


Eigen::MatrixXd doubleShiftMatrixProblem();

Eigen::MatrixXcd wilkinsonShiftMatrixProblem();

Eigen::MatrixXd Multiple0evProblem();

Eigen::MatrixXd ZeroMatrixProblem();

Eigen::MatrixXd SimilarEvs();

void run_small_tests();

#endif
