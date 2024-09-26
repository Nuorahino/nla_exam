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

  tridiagonal_matrix_nested<double> mat;
  mat.data.resize(size);
  CreateStdRandomTri(mat, seed);

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < 10; ++i) {
    std::vector<std::complex<double>> estimate = nla_exam::QrMethod<true>(mat, 1e-12);
  }
  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> runtime = (end - start);
  std::cout << "Runtime: " << runtime.count() << std::endl;

  return 1;
}
