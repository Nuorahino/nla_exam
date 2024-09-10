#include <iostream>
#include <complex>

#include "../../include/qr.hh"
#include "../../matrix/matrix_classes.hh"
#include "../createMatrix.hh"


// Maybe use std::optional instead
int main(int argc, char** argv) {
  std::cout << "Start Various Versions: " << std::endl;

  int size = 1000;
  int seed = 28;
  if(argc == 3) {
    size = atoi(argv[1]);
    seed = atoi(argv[2]);
  }

  tridiagonal_matrix2<double> mat;
  mat.diag.resize(size);
  mat.sdiag.resize(size - 1);
  CreateStdRandomVector(mat.diag, seed);
  CreateStdRandomVector(mat.sdiag, seed);

  //auto start = std::chrono::steady_clock::now();
  std::vector<std::complex<double>> estimate = nla_exam::QrMethod<true>(mat, 1e-12);
  //auto end = std::chrono::steady_clock::now();
  //std::chrono::duration<double> runtime = (end - start);

  return 1;
}
