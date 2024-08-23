#include <complex>

#include <catch2/catch_all.hpp>
#include <eigen3/Eigen/Dense>

#include "../include/qr.hh"
#include "../lapack/lapack_interface_impl.hh"
#include "helpfunctions_for_test.hh"


//TEMPLATE_TEST_CASE("Apply Givens Transformation", "[GivensRotation][BENCHMARK]", float, double, std::complex<float>, std::complex<double>) {
//  int n = GENERATE(take(5, random<int>(10, 100)));
//  Eigen::Matrix<TestType, -1, -1> A = Eigen::Matrix<TestType, -1, -1>::Random(n, n);
//  nla_exam::HessenbergTransformation(A);
//  BENCHMARK("Non Symmetric Givens Transformation of size " + std::to_string(n)) {
//    std::vector<TestType> givens_param = nla_exam::GetGivensEntries<TestType>(A(1,0), A(0,0));
//    nla_exam::ApplyGivensRight(A, givens_param.at(0), givens_param.at(1),
//                               givens_param.at(2));
//    nla_exam::ApplyGivensLeft(A, givens_param.at(0), givens_param.at(1),
//                              givens_param.at(2));
//  };
//}
//
//
//TEMPLATE_TEST_CASE("Apply Symmetric Givens Transformation", "[GivensRotation][BENCHMARK]", float, double, std::complex<float>, std::complex<double>) {
//  int n = GENERATE(take(5, random<int>(10, 100)));
//  Eigen::Matrix<TestType, -1, -1> A = Eigen::Matrix<TestType, -1, -1>::Random(n, n);
//  A = A + A.transpose().eval();
//  nla_exam::HessenbergTransformation(A, true);
//  BENCHMARK("Symmetric Givens Transformation of size " + std::to_string(n)) {
//    std::vector<TestType> givens_param = nla_exam::GetGivensEntries<TestType>(A(1,0), A(0,0));
//    nla_exam::ApplyGivensRight(A, givens_param.at(0), givens_param.at(1),
//                               givens_param.at(2));
//    nla_exam::ApplyGivensLeft(A, givens_param.at(0), givens_param.at(1),
//                              givens_param.at(2));
//  };
//}


TEMPLATE_TEST_CASE("Symmetric Hessenberg Transformation", "[GivensRotation][BENCHMARK]", float, double, std::complex<float>, std::complex<double>) {
  int n = GENERATE(take(5, random<int>(10, 100)));
  Eigen::Matrix<TestType, -1, -1> A = Eigen::Matrix<TestType, -1, -1>::Random(n, n);
  A = A + A.transpose().eval();
  BENCHMARK("Symmetric Hessenberg Transformation of size " + std::to_string(n)) {
  nla_exam::HessenbergTransformation(A, true);
  };
}


TEMPLATE_TEST_CASE("Non Symmetric Hessenberg Transformation", "[GivensRotation][BENCHMARK]", float, double, std::complex<float>, std::complex<double>) {
  int n = GENERATE(take(5, random<int>(10, 100)));
  Eigen::Matrix<TestType, -1, -1> A = Eigen::Matrix<TestType, -1, -1>::Random(n, n);
  A = A + A.transpose().eval();
  BENCHMARK("Non Symmetric Hessenberg Transformation of size " + std::to_string(n)) {
    nla_exam::HessenbergTransformation(A);
  };
}


TEMPLATE_TEST_CASE(
    "Implicit QR Step with exceptional shift Benchmark",
    "[ImplicitQRStep][BENCHMARK]", float, double, std::complex<float>,
    std::complex<double>) {
  int n = GENERATE(take(5, random<int>(10, 100)));
  Eigen::Matrix<TestType, -1, -1> A = Eigen::Matrix<TestType, -1, -1>::Random(n, n);
  nla_exam::HessenbergTransformation(A);

  BENCHMARK("Symmetric Implicit QR Step of size " + std::to_string(n)) {
    nla_exam::ImplicitQrStep<TestType, false>(A, false);
  };
}


TEMPLATE_TEST_CASE(
    "Symmetric Implicit QR step Benchmark",
    "[ImplicitQRStep][BENCHMARK]", float, double) {
  int n = GENERATE(take(5, random<int>(10, 100)));
  Eigen::Matrix<TestType, -1, -1> A = Eigen::Matrix<TestType, -1, -1>::Random(n, n);
  A = A + A.adjoint().eval();
  nla_exam::HessenbergTransformation(A);

  BENCHMARK("Symmetric Implicit QR Step of size " + std::to_string(n)) {
    nla_exam::ImplicitQrStep<TestType, true>(A, false);
  };

}


TEMPLATE_TEST_CASE("Double Shift QR Benchmark",
                   "[DoubleShiftQRStep][BENCHMARK]", float, double) {
  int n = GENERATE(take(5, random<int>(10, 100)));
  Eigen::Matrix<TestType, -1, -1> A = Eigen::Matrix<TestType, -1, -1>::Random(n, n);
  nla_exam::HessenbergTransformation(A);

  BENCHMARK("Double Shift QR Step of size " + std::to_string(n)) {
    nla_exam::DoubleShiftQrStep(A, false);
  };

}
