#include <iostream>
#include <complex>

#include <eigen3/Eigen/Dense>

#include "../../include/qr.hh"
#include "../lapack/lapack_interface_impl.hh"

template<typename Matrix>
void
CreateStdRandomTri(const Matrix &a_mat, const int ak_seed = std::time(nullptr)) {
  Matrix &mat = const_cast<Matrix&>(a_mat);
  //std::srand(ak_seed);
  std::mt19937 rng(ak_seed);
  std::uniform_real_distribution<double> distribution(-1000, 1000);
  std::size_t n = mat.rows();
  for (int i = 0; i < n - 1; ++i) {
    mat(i, i) = distribution(rng);
    mat(i, i + 1) = distribution(rng);
    mat(i + 1, i) = mat(i, i + 1);
  }
  mat(n - 1, n - 1) = distribution(rng);
  return;
}


// Maybe use std::optional instead
int main(int argc, char** argv) {
  std::cout << "Start Various Versions: " << std::endl;

  int size = 1000;
  int seed = 28;
  if(argc == 3) {
    size = atoi(argv[1]);
    seed = atoi(argv[2]);
  }

  Eigen::MatrixXd test(size, size);
  CreateStdRandomTri(test);

  auto start = std::chrono::steady_clock::now();
  int n = 10;
  for (int i = 0; i < n; ++i) {
    std::vector<std::complex<double>> estimate = nla_exam::QrMethod<true>(test, 1e-12);
  }
  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> runtime = (end - start);
  std::cout << "Runtime for: " << n << " is: " << runtime.count() << std::endl;

  return 0;
}
