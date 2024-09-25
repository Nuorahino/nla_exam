#include <iostream>
#include <complex>

#include <eigen3/Eigen/Dense>

#include "../../include/qr.hh"
#include "../../matrix/matrix_classes.hh"
#include "../createMatrix.hh"
#include "../../lapack/lapack_interface_impl.hh"


// Maybe use std::optional instead
int main(int argc, char** argv) {
  std::cout << "Start Various Versions: " << std::endl;

  int size = 1000;
  int seed = 28;
  if(argc == 3) {
    size = atoi(argv[1]);
    seed = atoi(argv[2]);
  }
  Eigen::MatrixXcd M = CreateStdRandomComplex<Eigen::MatrixXcd>(size, seed);
  //Eigen::MatrixXd M = CreateStdRandom<Eigen::MatrixXd>(size, seed);
  M = M + M.adjoint().eval();
//  Eigen::MatrixXd M_copy = M;
////  Eigen::MatrixXd M_lapack = M;
//  nla_exam::HessenbergTransformation<>(M, true);
//  std::cout << M_copy << std::endl;
//  auto start = std::chrono::steady_clock::now();
  nla_exam::HessenbergTransformation<true, true>(M, true);
//  //nla_exam::HessenbergTransformation<>(M, false);
//  auto end = std::chrono::steady_clock::now();
//  std::chrono::duration<double> runtime = (end - start);
//  std::cout << "Runtime: " << runtime.count() << std::endl;
  std::cout << M << std::endl;


//  std::cout << "Eigen" << std::endl;
//  start = std::chrono::steady_clock::now();
//  Eigen::HessenbergDecomposition<Eigen::MatrixXd> hd;
//  hd.compute(M_copy);
//  end = std::chrono::steady_clock::now();
//  runtime = (end - start);
//  std::cout << "Runtime Eigen: " << runtime.count() << std::endl;
//
//  std::cout << "LAPACK" << std::endl;
//  start = std::chrono::steady_clock::now();
//  Eigen::MatrixXd H = CreateHessenberg(M_lapack);
//  //Eigen::MatrixXd H = CreateTridiagonal(M_lapack);
//  end = std::chrono::steady_clock::now();
//  runtime = (end - start);
//  std::cout << "Runtime LAPACK: " << runtime.count() << std::endl;
//  //std::cout << hd.matrixH() << std::endl;


//  tridiagonal_matrix_nested<double> mat{M};
//
//  for (int i = 0; i < 10; ++i) {
//    std::vector<std::complex<double>> estimate = nla_exam::QrMethod<true>(mat, 1e-12);
//  }

  return 1;
}
