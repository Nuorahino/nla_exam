#include <iostream>

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#include "qr.hh"

void test_matrix(Eigen::MatrixXd& A) {
  std::cout << A << std::endl;
  //Eigen::VectorXd qr_res = qr_method(A);
  qr_method(A);

  Eigen::EigenSolver<Eigen::MatrixXd> es(A);
  Eigen::VectorXd eigen_res = es.eigenvalues().real();
  //std::cout << "qr: " << qr_res << std::endl;
  std::cout << "eigen: " << eigen_res << std::endl;
//  Eigen::MatrixXd Q = hessenberg_transformation(A, false);
//  std::cout << A << std::endl;
//  std::cout << Q * A * Q.transpose() << std::endl;
}

int main(int argc, char** argv) {
  std::cout << "Beginning of test" << std::endl;

  Eigen::MatrixXd A = Eigen::MatrixXd::Random(4,4);
  A = A + A.transpose().eval();
  test_matrix(A);
  Eigen::MatrixXd B = Eigen::MatrixXd::Random(4,4);
  test_matrix(B);

  return 1;
}
