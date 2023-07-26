#ifndef TEST_CAS_HH
#define TEST_CAS_HH

#include <eigen3/Eigen/Dense>


Eigen::MatrixXd doubleShiftMatrixProblem() {
  return Eigen::MatrixXd{{2, -1, 0}, {1, 2, -1}, {0, 1, 2}};
}

Eigen::MatrixXcd wilkinsonShiftMatrixProblem() {
  return Eigen::MatrixXd{{0.0, 0.0, 1.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0}};
}

Eigen::MatrixXd Multiple0evProblem() {
  return Eigen::MatrixXd{{2, 0, 0}, {1, 0, 0}, {0, 3, 0}};
}

Eigen::MatrixXd ZeroMatrixProblem() {
  //return Eigen::MatrixXd{{1e-16, 10e-15, 6e-15}, {-2e-14, -1e-22, -7e-15}, {0, -1e-22, 2e-16}};
  return Eigen::MatrixXd{{2e-16, 1e-17, 0}, {1e-15, 2e-16, -1e-22}, {0, -1e-42, 2e-16}};
}

#endif
