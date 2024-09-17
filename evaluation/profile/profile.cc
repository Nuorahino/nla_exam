#include <iostream>
#include <complex>

#include <blaze/math/DynamicMatrix.h>

#include "../../include/qr.hh"
#include "../../matrix/matrix_classes.hh"
#include "../createMatrix.hh"
#include "../lapack/lapack_interface_impl.hh"
#include <armadillo>


// Maybe use std::optional instead
int main(int argc, char** argv) {
  std::cout << "Start Various Versions: " << std::endl;

  int size = 1000;
  int seed = 28;
  if(argc == 3) {
    size = atoi(argv[1]);
    seed = atoi(argv[2]);
  }

  //tridiagonal_matrix2<double> mat;
  //mat.diag.resize(size);
  //mat.sdiag.resize(size - 1);
  //CreateStdRandomTri(mat);
  arma::Mat<std::complex<double>> mat(size, size, arma::fill::randu);

  auto start = std::chrono::steady_clock::now();
  int n = 10;
  for (int i = 0; i < n; ++i) {
    //std::vector<std::complex<double>> estimate = nla_exam::QrMethod<true>(mat, 1e-17);
    std::vector<std::complex<double>> estimate = nla_exam::QrMethod<false>(mat, 1e-12);
  }
  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> runtime = (end - start);
  std::cout << "Runtime for: " << n << " is: " << runtime.count() << std::endl;

  return 0;
}
