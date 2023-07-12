#ifndef TEST_CAS_HH
#define TEST_CAS_HH

#include <eigen3/Eigen/Dense>


Eigen::MatrixXd doubleShiftMatrixProblem() {
  return Eigen::MatrixXd{{2, -1, 0}, {1, 2, -1}, {0, 1, 2}};
}

Eigen::MatrixXcd wilkinsonShiftMatrixProblem() {
  return Eigen::MatrixXd{{0.0, 0.0, 1.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0}};
}
#endif
