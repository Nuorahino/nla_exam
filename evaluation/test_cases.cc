#include "test_cases.hh"

#include <iostream>
#include "qr.hh"
#include "helpfunctions/createMatrix.hh"

Eigen::MatrixXd doubleShiftMatrixProblem() {
  return Eigen::MatrixXd{{2, -1, 0}, {1, 2, -1}, {0, 1, 2}};
}

Eigen::MatrixXd wilkinsonShiftMatrixProblem() {
  return Eigen::MatrixXd{{0.0, 0.0, 1.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0}};
}

// Not diagonalizable
Eigen::MatrixXd Multiple0evProblem() {
  return Eigen::MatrixXd{{2, 0, 0}, {1, 1e-22, 0}, {0, 3, 0}};
  //return Eigen::MatrixXd{{-1e-19, 0, 0}, {2, 0, 0}, {0, 3, 1e-18}};
}

Eigen::MatrixXd ZeroMatrixProblem() {
  //return Eigen::MatrixXd{{1e-16, 10e-15, 0}, {-2e-14, -1e-22, -7e-15}, {0, -1e-22, 0}};
  return Eigen::MatrixXd{{2e-16, 1e-17, 0}, {1e-15, 2e-16, -1e-22}, {0, -1e-22, 2e-16}};
  //return Eigen::MatrixXd{{1e-16, 1e-17, 0}, {1e-15, 2e-16, -1e-22}, {0, -1e-22, 1e-16}};
}

Eigen::MatrixXd SimilarEvs() {
  Eigen::MatrixXd test{{-284.8, 59.0108, 0,0},{59.0108,-284.8, 0, 0}, {0,0, 592.886, 936.822}, {0,0, 936.822, 592.886}};

  Eigen::MatrixXd Q = CreateUnitaryMatrix<Eigen::MatrixXd>(4, 14);
  test = Q.adjoint() * test * Q;
    return test;
}

void run_small_tests() {
  std::vector<std::complex<double>> estimate;
  Eigen::EigenSolver<Eigen::MatrixXd> es;
  Eigen::ComplexEigenSolver<Eigen::MatrixXcd> esc;

  Eigen::MatrixXd test1 = doubleShiftMatrixProblem();
  estimate = nla_exam::QrMethod<false>(test1);
  std::cout << "test1: " << std::endl;
  std::cout << estimate << std::endl << std::endl;
  es.compute(test1);
  std::cout << es.eigenvalues() << std::endl;

  Eigen::MatrixXd test2 = wilkinsonShiftMatrixProblem();
  estimate = nla_exam::QrMethod<false>(test2);
  std::cout << "test2: " << std::endl;
  std::cout << estimate << std::endl << std::endl;;
  esc.compute(test2);
  std::cout << esc.eigenvalues() << std::endl;

  Eigen::MatrixXcd test3 = Multiple0evProblem();
  estimate = nla_exam::QrMethod<false>(test3, 1e-4);
  std::cout << "test3: " << std::endl;
  std::cout << estimate << std::endl << std::endl;
  esc.compute(test3);
  std::cout << esc.eigenvalues() << std::endl;

  Eigen::MatrixXd test4 = ZeroMatrixProblem();
  estimate = nla_exam::QrMethod<true>(test4);
  std::cout << "test4: " << std::endl;
  std::cout << estimate << std::endl << std::endl;
  es.compute(test4);
  std::cout << es.eigenvalues() << std::endl;

  Eigen::MatrixXd test5 = SimilarEvs();
  nla_exam::HessenbergTransformation<>(test5, true);
  estimate = nla_exam::QrMethod<false>(test5, 1e-12);
  std::cout << "test5: " << std::endl;
  std::cout << estimate << std::endl << std::endl;
  es.compute(test5);
  std::cout << es.eigenvalues() << std::endl;
}
