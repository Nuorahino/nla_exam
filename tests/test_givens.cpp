#include <complex>

#include <catch2/catch_all.hpp>
#include <eigen3/Eigen/Dense>
#include <complex.h>

#include "../include/qr.hh"
#include "../lapack/lapack_interface_impl.hh"
#include "helpfunctions_for_test.hh"


TEMPLATE_TEST_CASE("Arithmetic Givens Parameter for first parameter = 0 is correct", "[givens][givens_parameter]", float, double, std::complex<float>, std::complex<double>) {
  TestType b = GENERATE(take(5, ComplexRandom<TestType>(-100, 100)));
  TestType a = 0;
  auto res = nla_exam::GetGivensEntries(a, b);
  REQUIRE(res.at(0) == a);
  REQUIRE(res.at(1) == TestType{1});
}

//TEMPLATE_TEST_CASE("Arithmetic Givens Parameter for second parameter = 0 is correct", "[givens][givens_parameter]", int, float, double, std::complex<int>, std::complex<float>, std::complex<double>) {
TEMPLATE_TEST_CASE("Arithmetic Givens Parameter for second parameter = 0 is correct", "[givens][givens_parameter]", float, double, std::complex<float>, std::complex<double>) {
  TestType a = GENERATE(take(5, filter([] (TestType i) {return i != TestType{0};}, ComplexRandom<TestType>(-100, 100))));
  TestType b = 0;

  auto res = nla_exam::GetGivensEntries(a, b);
  REQUIRE(res.at(0) == TestType{1});
  REQUIRE(res.at(1) == b);
}

TEMPLATE_TEST_CASE("Givens Vector is of length one", "[givens][givens_parameter]", float, double, std::complex<float>, std::complex<double>) {
  TestType a = GENERATE(take(5, ComplexRandom<TestType>(-100, 100)));
  TestType b = GENERATE(take(5, ComplexRandom<TestType>(-100, 100)));
  auto res = nla_exam::GetGivensEntries(a, b);
  REQUIRE_THAT(std::hypot(std::abs(res.at(0)), std::abs(res.at(1))), Catch::Matchers::WithinAbs(1, tol<TestType>()));
}

TEMPLATE_TEST_CASE("Compare Givens Parameter with LAPACK", "[givens][givens_parameter]", float, double, std::complex<float>, std::complex<double>) {
  TestType a = GENERATE(take(5, ComplexRandom<TestType>(-100, 100)));
  TestType b = GENERATE(take(5, ComplexRandom<TestType>(-100, 100)));
  std::vector<TestType> res_qr = nla_exam::GetGivensEntries<TestType>(a, b);
  std::vector<TestType> res_lapack = compute_givens_parameter<TestType>(a, b);
  REQUIRE_THAT(std::abs(res_qr.at(0) - res_lapack.at(0)), Catch::Matchers::WithinAbs(0.0, tol<TestType>()));
  REQUIRE_THAT(std::abs(res_qr.at(1) - res_lapack.at(1)), Catch::Matchers::WithinAbs(0.0, tol<TestType>()));
}

//TODO: This is wrong, use the Givens rotation instead
TEMPLATE_TEST_CASE("Right Givens Rotation does not change the Norm", "[givens]", float, double, std::complex<float>, std::complex<double>) {
  constexpr int n = 10;
  int i = GENERATE(take(5, random<int>(0, n-1)));
  int j = GENERATE(take(5, random<int>(0, n-1)));
  if ( i == j ) j = (i + 1) % (n - 1);
  Eigen::Matrix<TestType, -1, -1> A = Eigen::Matrix<TestType, -1, -1>::Random(n, n);
  Eigen::Matrix<TestType, -1, -1> A_copy = A;
  std::vector<TestType> param = nla_exam::GetGivensEntries(A(0, 0), A(0, 1));
  nla_exam::ApplyGivensRight(A(Eigen::all, {i, j}), param.at(0), param.at(1), param.at(2));
  REQUIRE_THAT(A.norm(), Catch::Matchers::WithinAbs(A_copy.norm(), tol<TestType>() * 1e1));
}

TEMPLATE_TEST_CASE("Left Givens Rotation does not change the Norm", "[givens]", float, double, std::complex<float>, std::complex<double>) {
  constexpr int n = 10;
  int i = GENERATE(take(5, random<int>(0, n-1)));
  int j = GENERATE(take(5, random<int>(0, n-1)));
  if ( i == j ) j = (i + 1) % (n - 1);
  Eigen::Matrix<TestType, -1, -1> A = Eigen::Matrix<TestType, -1, -1>::Random(n, n);
  Eigen::Matrix<TestType, -1, -1> A_copy = A;
  std::vector<TestType> param = nla_exam::GetGivensEntries(A(0,0), A(1,0));
  nla_exam::ApplyGivensLeft(A({i,j}, Eigen::all), param.at(0), param.at(1), param.at(2));
  REQUIRE_THAT(A.norm(), Catch::Matchers::WithinAbs(A_copy.norm(), tol<TestType>() * 1e1));
}

TEMPLATE_TEST_CASE("Givens Parameter combined with applying the left givens rotation using LAPACK sets the entry to 0", "[givens][givens_parameter]", float, double, std::complex<float>, std::complex<double>) {
  int n = GENERATE(take(5, random<int>(3, 10)));
  Eigen::Matrix<TestType, -1, -1> A = Eigen::Matrix<TestType, -1, -1>::Random(n, n);
  std::vector<TestType> param = nla_exam::GetGivensEntries(A(0,0), A(1,0));
  Eigen::Matrix<TestType, -1, -1> res = apply_givens_left<TestType>(A, 0, get_real(param.at(0)), param.at(1));
  REQUIRE_THAT(std::abs(res(1,0)), Catch::Matchers::WithinAbs(0.0, tol<TestType>()));
}

TEMPLATE_TEST_CASE("Givens Parameter combined with applying the right givens rotation using LAPACK sets the entry to 0", "[givens][givens_parameter]", float, double, std::complex<float>, std::complex<double>) {
  int n = GENERATE(take(5, random<int>(3, 10)));
  Eigen::Matrix<TestType, -1, -1> A = Eigen::Matrix<TestType, -1, -1>::Random(n, n);
  std::vector<TestType> param = nla_exam::GetGivensEntries(A(0,0), A(0,1));
  Eigen::Matrix<TestType, -1, -1> res = apply_givens_right<TestType>(A, 0, get_real(param.at(0)), complex_conj(param.at(1)));
  REQUIRE_THAT(std::abs(res(0,1)), Catch::Matchers::WithinAbs(0.0, tol<TestType>()));
}

TEMPLATE_TEST_CASE("Left Givens Rotation has same result as LAPACK", "[givens][apply_givens]", float, double, std::complex<float>, std::complex<double>) {
  constexpr int n = 20;
  int i = GENERATE(take(5, random<int>(0, n-1)));
  int j = GENERATE(take(5, random<int>(0, n-1)));
  if ( i == j ) j = (i + 1) % (n - 1);
  Eigen::Matrix<TestType, -1, -1>  A = Eigen::Matrix<TestType, -1, -1>::Random(n, n);
  std::vector<TestType> param = nla_exam::GetGivensEntries(A(0,0), A(1,0));
  Eigen::Matrix<TestType, -1, -1> res = apply_givens_left<TestType>(A, i, j, get_real(param.at(0)), param.at(1));
  nla_exam::ApplyGivensLeft(A({i,j}, Eigen::all), param.at(0), param.at(1), param.at(2));
  REQUIRE_THAT((A - res).norm(), Catch::Matchers::WithinAbs(0.0, tol<TestType>()));
}

// This is a Problem atm
TEMPLATE_TEST_CASE("Right Givens Rotation has same result as LAPACK", "[givens][apply_givens]", float, double, std::complex<float>, std::complex<double>){
  typedef Eigen::Matrix<TestType, -1, -1> Mat;
  constexpr int n = 20;
  int i = GENERATE(take(5, random<int>(0, n-1)));
  int j = GENERATE(take(5, random<int>(0, n-1)));
  if ( i == j ) j = (i + 1) % (n - 1);
  Mat A = Mat::Random(n, n);
  std::vector<TestType> param = nla_exam::GetGivensEntries(A(0,0), A(0,1));
  Mat res = apply_givens_right<TestType>(A, i, j, get_real(param.at(0)), param.at(1));
  nla_exam::ApplyGivensRight(A(Eigen::all, {i, j}), param.at(0), param.at(1), param.at(2));
  REQUIRE_THAT((A - res).norm(), Catch::Matchers::WithinAbs(0.0, tol<TestType>()));
}

TEMPLATE_TEST_CASE("Real Right Givens Rotation has same result in real and complex function", "[givens][apply_givens]", float, double) {
  constexpr int n = 20;
  int i = GENERATE(take(5, random<int>(0, n-1)));
  int j = GENERATE(take(5, random<int>(0, n-1)));
  if ( i == j ) j = (i + 1) % (n - 1);
  Eigen::Matrix<TestType, -1, -1> A = Eigen::Matrix<TestType, -1, -1>::Random(n, n);
  Eigen::Matrix<TestType, -1, -1> A_copy = A;
  std::vector<TestType> param = nla_exam::GetGivensEntries(A(0,0), A(1,0));
  nla_exam::ApplyGivensRight(A(Eigen::all, {i,j}), param.at(0), param.at(1), param.at(1));
  nla_exam::ApplyGivensRight(A_copy(Eigen::all, {i,j}), param.at(0), param.at(1));
  REQUIRE_THAT((A - A_copy).norm(), Catch::Matchers::WithinAbs(0.0, tol<TestType>()));
}

TEMPLATE_TEST_CASE("Real Left Givens Rotation has same result in real and complex function", "[givens][apply_givens]", float, double) {
  constexpr int n = 20;
  int i = GENERATE(take(5, random<int>(0, n-1)));
  int j = GENERATE(take(5, random<int>(0, n-1)));
  if ( i == j ) j = (i + 1) % (n - 1);
  Eigen::Matrix<TestType, -1, -1> A = Eigen::Matrix<TestType, -1, -1>::Random(n, n);
  Eigen::Matrix<TestType, -1, -1> A_copy = A;
  std::vector<TestType> param = nla_exam::GetGivensEntries(A(0,0), A(0,1));
  nla_exam::ApplyGivensLeft(A({i,j}, Eigen::all), param.at(0), param.at(1), param.at(1));
  nla_exam::ApplyGivensLeft(A_copy({i,j}, Eigen::all), param.at(0), param.at(1));
  REQUIRE_THAT((A - A_copy).norm(), Catch::Matchers::WithinAbs(0.0, tol<TestType>()));
}

TEMPLATE_TEST_CASE("Right Givens Rotation does not change other columns", "[givens][apply_givens]", float, double, std::complex<float>, std::complex<double>) {
  constexpr int n = 20;
  int i = GENERATE(take(5, random<int>(0, n-1)));
  int j = GENERATE(take(5, random<int>(0, n-1)));
  if ( i == j ) j = (i + 1) % (n - 1);
  Eigen::Matrix<TestType, -1, -1> A = Eigen::Matrix<TestType, -1, -1>::Random(n, n);
  Eigen::Matrix<TestType, -1, -1> A_copy = A;
  std::vector<TestType> param = nla_exam::GetGivensEntries(A(0,0), A(1,0));
  nla_exam::ApplyGivensRight(A(Eigen::all, {i,j}), param.at(0), param.at(1), param.at(1));
  A(Eigen::all, {i,j}).setZero();
  A_copy(Eigen::all, {i,j}).setZero();
  REQUIRE_THAT((A - A_copy).norm(), Catch::Matchers::WithinAbs(0.0, tol<TestType>()));
}

TEMPLATE_TEST_CASE("Left Givens Rotation does not change other columns", "[givens][apply_givens]", float, double, std::complex<float>, std::complex<double>) {
  constexpr int n = 20;
  int i = GENERATE(take(5, random<int>(0, n-1)));
  int j = GENERATE(take(5, random<int>(0, n-1)));
  if ( i == j ) j = (i + 1) % (n - 1);
  Eigen::Matrix<TestType, -1, -1> A = Eigen::Matrix<TestType, -1, -1>::Random(n, n);
  Eigen::Matrix<TestType, -1, -1> A_copy = A;
  std::vector<TestType> param = nla_exam::GetGivensEntries(A(0,0), A(1,0));
  nla_exam::ApplyGivensLeft(A({i,j}, Eigen::all), param.at(0), param.at(1), param.at(1));
  A({i,j}, Eigen::all).setZero();
  A_copy({i,j}, Eigen::all).setZero();
  REQUIRE_THAT((A - A_copy).norm(), Catch::Matchers::WithinAbs(0.0, tol<TestType>()));
}
