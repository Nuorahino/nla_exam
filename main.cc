#include <iostream>
#include <complex>

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#include "qr.hh"

void test_matrix(Eigen::MatrixXd& A) {
  std::cout << A << std::endl;

  Eigen::EigenSolver<Eigen::MatrixXd> es(A);
  Eigen::VectorXd eigen_res = es.eigenvalues().real();
  std::cout << "eigen: " << eigen_res << std::endl;
  std::vector<std::complex<double>> qr_res = qr_method(A);
  //qr_method(A);
  std::cout << "qr: " << qr_res << std::endl;
}

int main(int argc, char** argv) {
  std::cout << "Beginning of test" << std::endl;

  Eigen::MatrixXd A = Eigen::MatrixXd::Random(8,8);
  A = A + A.transpose().eval();
  test_matrix(A);
  Eigen::MatrixXd B = Eigen::MatrixXd::Random(8,8);
  test_matrix(B);

  return 1;
}
