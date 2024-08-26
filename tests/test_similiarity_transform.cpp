#include <iostream>

#include <catch2/catch_all.hpp>
#include <eigen3/Eigen/Dense>

#include "../include/qr.hh"
#include "../lapack/lapack_interface_impl.hh"
#include "helpfunctions_for_test.hh"


TEMPLATE_TEST_CASE("Non Symmetric Hessenberg Transformation has correctform",
                   "[HessenbergTransformation]", float, double, std::complex<float>,
                   std::complex<double>) {
  int n = GENERATE(take(5, random<int>(3, 20)));
  Eigen::Matrix<TestType, -1, -1> A = Eigen::Matrix<TestType, -1, -1>::Random(n, n);
  nla_exam::HessenbergTransformation(A);
  for (int i = -2; i > -n; --i) {
    REQUIRE_THAT(A.diagonal(i).norm(), Catch::Matchers::WithinAbs(0.0, tol<TestType>()));
  }
}


TEMPLATE_TEST_CASE("Non Symmetric Hessenberg Transformation does not change the norm",
                   "[HessenbergTransformation]", float, double, std::complex<float>,
                   std::complex<double>) {
  int n = GENERATE(take(5, random<int>(3, 20)));
  Eigen::Matrix<TestType, -1, -1> A = Eigen::Matrix<TestType, -1, -1>::Random(n, n);
  Eigen::Matrix<TestType, -1, -1> A_copy = A;
  nla_exam::HessenbergTransformation(A);
  REQUIRE_THAT(A.norm(),
               Catch::Matchers::WithinAbs(A_copy.norm(), tol<TestType>() * 1e1));
}


TEMPLATE_TEST_CASE(
    "Non Symmetric Hessenberg Transformation does not change the eigenvalues",
    "[HessenbergTransformation]", float, double, std::complex<float>,
    std::complex<double>) {
  int n = GENERATE(take(5, random<int>(3, 40)));
  Eigen::Matrix<TestType, -1, -1> A = Eigen::Matrix<TestType, -1, -1>::Random(n, n);
  Eigen::Matrix<TestType, -1, -1> eigenvector;
  Eigen::Vector<typename ComplexDataType<TestType>::type, -1> eigenvalues(n);
  Eigen::Vector<typename ComplexDataType<TestType>::type, -1> new_eigenvalues(n);

  std::tie(eigenvector, eigenvalues) = CalculateGeneralEigenvalues<TestType>(A, false);
  nla_exam::HessenbergTransformation(A);
  std::tie(eigenvector, new_eigenvalues) =
      CalculateGeneralEigenvalues<TestType>(A, false);
  std::vector<typename ComplexDataType<TestType>::type> ev(eigenvalues.data(),
                                                           eigenvalues.data() + n);
  std::vector<typename ComplexDataType<TestType>::type> nev(new_eigenvalues.data(),
                                                            new_eigenvalues.data() + n);
  order_as_min_matching(ev, nev);
  for (unsigned int i = 0; i < ev.size(); ++i) {
    REQUIRE_THAT(std::abs(ev.at(i) - nev.at(i)),
                 Catch::Matchers::WithinAbs(0.0, tol<TestType>() * 1e2));
  }
}


TEMPLATE_TEST_CASE("Symmetric Hessenberg Transformation does not change the eigenvalues",
                   "[HessenbergTransformation]", float, double, std::complex<float>,
                   std::complex<double>) {
  int n = GENERATE(take(5, random<int>(3, 40)));
  Eigen::Matrix<TestType, -1, -1> A = Eigen::Matrix<TestType, -1, -1>::Random(n, n);
  A = A + A.adjoint().eval();
  Eigen::Matrix<TestType, -1, -1> eigenvector;
  Eigen::Vector<typename ComplexDataType<TestType>::type, -1> eigenvalues(n);
  Eigen::Vector<typename ComplexDataType<TestType>::type, -1> new_eigenvalues(n);

  std::tie(eigenvector, eigenvalues) = CalculateGeneralEigenvalues<TestType>(A, false);
  nla_exam::HessenbergTransformation(A, true);
  std::tie(eigenvector, new_eigenvalues) =
      CalculateGeneralEigenvalues<TestType>(A, false);
  std::vector<typename ComplexDataType<TestType>::type> ev(eigenvalues.data(),
                                                           eigenvalues.data() + n);
  std::vector<typename ComplexDataType<TestType>::type> nev(new_eigenvalues.data(),
                                                            new_eigenvalues.data() + n);
  order_as_min_matching(ev, nev);
  for (unsigned int i = 0; i < ev.size(); ++i) {
    REQUIRE_THAT(std::abs(ev.at(i) - nev.at(i)),
                 Catch::Matchers::WithinAbs(0.0, tol<TestType>() * 1e2));
  }
}

TEMPLATE_TEST_CASE("Non Symmetric Householder Reflection does not change the eigenvalues",
                   "[HessenbergTransformation]", float, double, std::complex<float>,
                   std::complex<double>) {
  int n = GENERATE(take(5, random<int>(3, 40)));
  Eigen::Matrix<TestType, -1, -1> A = Eigen::Matrix<TestType, -1, -1>::Random(n, n);
  Eigen::Vector<TestType, -1> v = Eigen::Vector<TestType, -1>::Random(n);
  Eigen::Matrix<TestType, -1, -1> eigenvector;
  Eigen::Vector<typename ComplexDataType<TestType>::type, -1> eigenvalues(n);
  Eigen::Vector<typename ComplexDataType<TestType>::type, -1> new_eigenvalues(n);

  std::tie(eigenvector, eigenvalues) = CalculateGeneralEigenvalues<TestType>(A, false);
  Eigen::Vector<TestType, -1> w = nla_exam::GetHouseholderVector(v);
  nla_exam::ApplyHouseholderRight(w, A);
  nla_exam::ApplyHouseholderLeft(w, A);
  std::tie(eigenvector, new_eigenvalues) =
      CalculateGeneralEigenvalues<TestType>(A, false);
  std::vector<typename ComplexDataType<TestType>::type> ev(eigenvalues.data(),
                                                           eigenvalues.data() + n);
  std::vector<typename ComplexDataType<TestType>::type> nev(new_eigenvalues.data(),
                                                            new_eigenvalues.data() + n);
  order_as_min_matching(ev, nev);
  for (unsigned int i = 0; i < ev.size(); ++i) {
    REQUIRE_THAT(std::abs(ev.at(i) - nev.at(i)),
                 Catch::Matchers::WithinAbs(0.0, tol<TestType>() * 1e2));
  }
}


TEMPLATE_TEST_CASE("Symmetric Householder Reflection does not change the eigenvalues",
                   "[HessenbergTransformation]", float, double, std::complex<float>,
                   std::complex<double>) {
  int n = GENERATE(take(5, random<int>(3, 40)));
  Eigen::Matrix<TestType, -1, -1> A = Eigen::Matrix<TestType, -1, -1>::Random(n, n);
  Eigen::Vector<TestType, -1> v = Eigen::Vector<TestType, -1>::Random(n);
  A = A + A.adjoint().eval();
  Eigen::Matrix<TestType, -1, -1> eigenvector;
  Eigen::Vector<typename ComplexDataType<TestType>::type, -1> eigenvalues(n);
  Eigen::Vector<typename ComplexDataType<TestType>::type, -1> new_eigenvalues(n);

  std::tie(eigenvector, eigenvalues) = CalculateGeneralEigenvalues<TestType>(A, false);
  Eigen::Vector<TestType, -1> w = nla_exam::GetHouseholderVector(v);
  nla_exam::ApplyHouseholderRight(w, A);
  nla_exam::ApplyHouseholderLeft(w, A);

  std::tie(eigenvector, new_eigenvalues) =
      CalculateGeneralEigenvalues<TestType>(A, false);
  std::vector<typename ComplexDataType<TestType>::type> ev(eigenvalues.data(),
                                                           eigenvalues.data() + n);
  std::vector<typename ComplexDataType<TestType>::type> nev(new_eigenvalues.data(),
                                                            new_eigenvalues.data() + n);
  order_as_min_matching(ev, nev);
  for (unsigned int i = 0; i < ev.size(); ++i) {
    REQUIRE_THAT(std::abs(ev.at(i) - nev.at(i)),
                 Catch::Matchers::WithinAbs(0.0, tol<TestType>() * 1e2));
  }
}


TEMPLATE_TEST_CASE("Non Symmetric Givens Rotation does not change the eigenvalues",
                   "[GivensRotation]", float, double, std::complex<float>,
                   std::complex<double>) {
  int n = GENERATE(take(5, random<int>(3, 40)));
  Eigen::Matrix<TestType, -1, -1> A = Eigen::Matrix<TestType, -1, -1>::Random(n, n);
  TestType a = GENERATE(take(5, ComplexRandom<TestType>(-100, 100)));
  TestType b = GENERATE(take(5, ComplexRandom<TestType>(-100, 100)));
  std::vector<TestType> givens_param = nla_exam::GetGivensEntries<TestType>(a, b);
  Eigen::Matrix<TestType, -1, -1> eigenvector;
  Eigen::Vector<typename ComplexDataType<TestType>::type, -1> eigenvalues(n);
  Eigen::Vector<typename ComplexDataType<TestType>::type, -1> new_eigenvalues(n);

  std::tie(eigenvector, eigenvalues) = CalculateGeneralEigenvalues<TestType>(A, false);
  nla_exam::ApplyGivensRight(A, givens_param.at(0), givens_param.at(1),
                             givens_param.at(2));
  nla_exam::ApplyGivensLeft(A, givens_param.at(0), givens_param.at(1),
                            givens_param.at(2));
  std::tie(eigenvector, new_eigenvalues) =
      CalculateGeneralEigenvalues<TestType>(A, false);
  std::vector<typename ComplexDataType<TestType>::type> ev(eigenvalues.data(),
                                                           eigenvalues.data() + n);
  std::vector<typename ComplexDataType<TestType>::type> nev(new_eigenvalues.data(),
                                                            new_eigenvalues.data() + n);
  order_as_min_matching(ev, nev);
  for (unsigned int i = 0; i < ev.size(); ++i) {
    REQUIRE_THAT(std::abs(ev.at(i) - nev.at(i)),
                 Catch::Matchers::WithinAbs(0.0, tol<TestType>() * 1e2));
  }
}


TEMPLATE_TEST_CASE("Symmetric Givens Rotation does not change the eigenvalues",
                   "[GivensRotation]", float, double, std::complex<float>,
                   std::complex<double>) {
  int n = GENERATE(take(5, random<int>(3, 40)));
  Eigen::Matrix<TestType, -1, -1> A = Eigen::Matrix<TestType, -1, -1>::Random(n, n);
  TestType a = GENERATE(take(5, ComplexRandom<TestType>(-100, 100)));
  TestType b = GENERATE(take(5, ComplexRandom<TestType>(-100, 100)));
  std::vector<TestType> givens_param = nla_exam::GetGivensEntries<TestType>(a, b);
  A = A + A.adjoint().eval();
  Eigen::Matrix<TestType, -1, -1> eigenvector;
  Eigen::Vector<typename ComplexDataType<TestType>::type, -1> eigenvalues(n);
  Eigen::Vector<typename ComplexDataType<TestType>::type, -1> new_eigenvalues(n);

  std::tie(eigenvector, eigenvalues) = CalculateGeneralEigenvalues<TestType>(A, false);

  std::tie(eigenvector, new_eigenvalues) =
      CalculateGeneralEigenvalues<TestType>(A, false);
  std::vector<typename ComplexDataType<TestType>::type> ev(eigenvalues.data(),
                                                           eigenvalues.data() + n);
  std::vector<typename ComplexDataType<TestType>::type> nev(new_eigenvalues.data(),
                                                            new_eigenvalues.data() + n);
  order_as_min_matching(ev, nev);
  for (unsigned int i = 0; i < ev.size(); ++i) {
    REQUIRE_THAT(std::abs(ev.at(i) - nev.at(i)),
                 Catch::Matchers::WithinAbs(0.0, tol<TestType>() * 1e2));
  }
}


TEMPLATE_TEST_CASE("Real Symmetric Wilkinson Shift is correct", "[Wilkinson]", float, double) {
  int n = 2;
  Eigen::Matrix<TestType, -1, -1> A = Eigen::Matrix<TestType, -1, -1>::Random(n, n);
  A = A + A.transpose().eval();
  Eigen::Matrix<TestType, -1, -1> eigenvector;
  Eigen::Vector<typename ComplexDataType<TestType>::type, -1> eigenvalues(n);
  std::tie(eigenvector, eigenvalues) = CalculateGeneralEigenvalues<TestType>(A, false);
  TestType shift = nla_exam::WilkinsonShift<TestType, true>(A);
  TestType res = eigenvalues(0).real();
  if(std::abs(eigenvalues(1).real() - A(1,1)) < std::abs(eigenvalues(0).real() - A(1,1))) {
    res = eigenvalues(1).real();
  }
  REQUIRE_THAT(std::abs(shift - res), Catch::Matchers::WithinAbs(0.0, tol<TestType>()));
}


TEMPLATE_TEST_CASE("Complex Symmetric Wilkinson Shift is correct", "[Wilkinson]", std::complex<float>, std::complex<double>) {
  int n = 2;
  Eigen::Matrix<TestType, -1, -1> A = Eigen::Matrix<TestType, -1, -1>::Random(n, n);
  A = A + A.transpose().eval();
  Eigen::Matrix<TestType, -1, -1> eigenvector;
  Eigen::Vector<typename ComplexDataType<TestType>::type, -1> eigenvalues(n);
  std::tie(eigenvector, eigenvalues) = CalculateGeneralEigenvalues<TestType>(A, false);
  TestType shift = nla_exam::WilkinsonShift<TestType, true>(A);
  TestType res = eigenvalues(0);
  if(std::abs(eigenvalues(1) - A(1,1)) < std::abs(eigenvalues(0) - A(1,1))) {
    res = eigenvalues(1);
  }
  REQUIRE_THAT(std::abs(shift - res), Catch::Matchers::WithinAbs(0.0, tol<TestType>()));
}


TEMPLATE_TEST_CASE("Complex General Wilkinson Shift is correct", "[Wilkinson]", std::complex<float>, std::complex<double>) {
  int n = 2;
  Eigen::Matrix<TestType, -1, -1> A = Eigen::Matrix<TestType, -1, -1>::Random(n, n);
  Eigen::Matrix<TestType, -1, -1> eigenvector;
  Eigen::Vector<typename ComplexDataType<TestType>::type, -1> eigenvalues(n);
  std::tie(eigenvector, eigenvalues) = CalculateGeneralEigenvalues<TestType>(A, false);
  TestType shift = nla_exam::WilkinsonShift<TestType, false>(A);
  TestType res = eigenvalues(0);
  if(std::abs(eigenvalues(1) - A(1,1)) < std::abs(eigenvalues(0) - A(1,1))) {
    res = eigenvalues(1);
  }
  REQUIRE_THAT(std::abs(shift - res), Catch::Matchers::WithinAbs(0.0, tol<TestType>()));
}


TEMPLATE_TEST_CASE("Implicit QR Step does not change the eigenvalues",
                   "[ImplicitQRStep]", std::complex<float>,
                   std::complex<double>) {
  int n = GENERATE(take(5, random<int>(3, 40)));
  Eigen::Matrix<TestType, -1, -1> A = Eigen::Matrix<TestType, -1, -1>::Random(n, n);
  nla_exam::HessenbergTransformation(A);

  Eigen::Matrix<TestType, -1, -1> eigenvector;
  Eigen::Vector<typename ComplexDataType<TestType>::type, -1> eigenvalues(n);
  Eigen::Vector<typename ComplexDataType<TestType>::type, -1> new_eigenvalues(n);

  std::tie(eigenvector, eigenvalues) = CalculateGeneralEigenvalues<TestType>(A, false);
  nla_exam::ImplicitQrStep<TestType, false>(A, false);

  std::tie(eigenvector, new_eigenvalues) =
      CalculateGeneralEigenvalues<TestType>(A, false);
  std::vector<typename ComplexDataType<TestType>::type> ev(eigenvalues.data(),
                                                           eigenvalues.data() + n);
  std::vector<typename ComplexDataType<TestType>::type> nev(new_eigenvalues.data(),
                                                            new_eigenvalues.data() + n);
  order_as_min_matching(ev, nev);
  for (unsigned int i = 0; i < ev.size(); ++i) {
    REQUIRE_THAT(std::abs(ev.at(i) - nev.at(i)),
                 Catch::Matchers::WithinAbs(0.0, tol<TestType>() * 1e2));
  }
}


TEMPLATE_TEST_CASE("Symmetric Implicit QR step does not change the eigenvalues",
                   "[ImplicitQRStep]", float, double) {
  int n = GENERATE(take(5, random<int>(3, 40)));
  Eigen::Matrix<TestType, -1, -1> A = Eigen::Matrix<TestType, -1, -1>::Random(n, n);
  A = A + A.adjoint().eval();
  nla_exam::HessenbergTransformation(A, true);

  Eigen::Matrix<TestType, -1, -1> eigenvector;
  Eigen::Vector<typename ComplexDataType<TestType>::type, -1> eigenvalues(n);
  Eigen::Vector<typename ComplexDataType<TestType>::type, -1> new_eigenvalues(n);

  std::tie(eigenvector, eigenvalues) = CalculateGeneralEigenvalues<TestType>(A, false);
  nla_exam::ImplicitQrStep<TestType, true, Eigen::Matrix<TestType, -1, -1>>(A, false);

  std::tie(eigenvector, new_eigenvalues) =
      CalculateGeneralEigenvalues<TestType>(A, false);
  std::vector<typename ComplexDataType<TestType>::type> ev(eigenvalues.data(),
                                                           eigenvalues.data() + n);
  std::vector<typename ComplexDataType<TestType>::type> nev(new_eigenvalues.data(),
                                                            new_eigenvalues.data() + n);
  order_as_min_matching(ev, nev);
  for (unsigned int i = 0; i < ev.size(); ++i) {
    REQUIRE_THAT(std::abs(ev.at(i) - nev.at(i)),
                 Catch::Matchers::WithinAbs(0.0, tol<TestType>() * 1e2));
  }
}


TEMPLATE_TEST_CASE(
    "Implicit QR Step with exceptional shift does not change the eigenvalues",
    "[ImplicitQRStep]", float, double, std::complex<float>,
    std::complex<double>) {
  int n = GENERATE(take(5, random<int>(3, 40)));
  Eigen::Matrix<TestType, -1, -1> A = Eigen::Matrix<TestType, -1, -1>::Random(n, n);
  nla_exam::HessenbergTransformation(A);

  Eigen::Matrix<TestType, -1, -1> eigenvector;
  Eigen::Vector<typename ComplexDataType<TestType>::type, -1> eigenvalues(n);
  Eigen::Vector<typename ComplexDataType<TestType>::type, -1> new_eigenvalues(n);

  std::tie(eigenvector, eigenvalues) = CalculateGeneralEigenvalues<TestType>(A, false);
  nla_exam::ImplicitQrStep<TestType, false>(A, true);

  std::tie(eigenvector, new_eigenvalues) =
      CalculateGeneralEigenvalues<TestType>(A, false);
  std::vector<typename ComplexDataType<TestType>::type> ev(eigenvalues.data(),
                                                           eigenvalues.data() + n);
  std::vector<typename ComplexDataType<TestType>::type> nev(new_eigenvalues.data(),
                                                            new_eigenvalues.data() + n);
  order_as_min_matching(ev, nev);
  for (unsigned int i = 0; i < ev.size(); ++i) {
    REQUIRE_THAT(std::abs(ev.at(i) - nev.at(i)),
                 Catch::Matchers::WithinAbs(0.0, tol<TestType>() * 1e2));
  }
}


TEMPLATE_TEST_CASE(
    "Symmetric Implicit QR step with exceptional shift does not change the eigenvalues",
    "[ImplicitQRStep]", float, double) {
  int n = GENERATE(take(5, random<int>(3, 40)));
  Eigen::Matrix<TestType, -1, -1> A = Eigen::Matrix<TestType, -1, -1>::Random(n, n);
  A = A + A.adjoint().eval();
  nla_exam::HessenbergTransformation(A, true);

  Eigen::Matrix<TestType, -1, -1> eigenvector;
  Eigen::Vector<typename ComplexDataType<TestType>::type, -1> eigenvalues(n);
  Eigen::Vector<typename ComplexDataType<TestType>::type, -1> new_eigenvalues(n);

  std::tie(eigenvector, eigenvalues) = CalculateGeneralEigenvalues<TestType>(A, false);
  nla_exam::ImplicitQrStep<TestType, true>(A, true);

  std::tie(eigenvector, new_eigenvalues) =
      CalculateGeneralEigenvalues<TestType>(A, false);
  std::vector<typename ComplexDataType<TestType>::type> ev(eigenvalues.data(),
                                                           eigenvalues.data() + n);
  std::vector<typename ComplexDataType<TestType>::type> nev(new_eigenvalues.data(),
                                                            new_eigenvalues.data() + n);
  order_as_min_matching(ev, nev);
  for (unsigned int i = 0; i < ev.size(); ++i) {
    REQUIRE_THAT(std::abs(ev.at(i) - nev.at(i)),
                 Catch::Matchers::WithinAbs(0.0, tol<TestType>() * 1e2));
  }
}


TEMPLATE_TEST_CASE("Double Shift is correct", "[DoubleShift]", float, double) {
  int n = 2;
  Eigen::Matrix<TestType, -1, -1> A = Eigen::Matrix<TestType, -1, -1>::Random(n, n);
  Eigen::Matrix<TestType, -1, -1> eigenvector;
  Eigen::Vector<typename ComplexDataType<TestType>::type, -1> eigenvalues(n);
  std::tie(eigenvector, eigenvalues) = CalculateGeneralEigenvalues<TestType>(A, false);
  std::vector<TestType> shift = nla_exam::DoubleShiftParameter(A);
  REQUIRE_THAT(std::abs(shift.at(0) + (eigenvalues(0) + eigenvalues(1))), Catch::Matchers::WithinAbs(0.0, tol<TestType>()));
  REQUIRE_THAT(std::abs(shift.at(1) - (eigenvalues(0) * eigenvalues(1))), Catch::Matchers::WithinAbs(0.0, tol<TestType>()));
}


TEMPLATE_TEST_CASE("Double Shift QR Step does not change the eigenvalues",
                   "[DoubleShiftQRStep]", float, double) {
  int n = GENERATE(take(5, random<int>(3, 40)));
  Eigen::Matrix<TestType, -1, -1> A = Eigen::Matrix<TestType, -1, -1>::Random(n, n);
  nla_exam::HessenbergTransformation(A);

  Eigen::Matrix<TestType, -1, -1> eigenvector;
  Eigen::Vector<typename ComplexDataType<TestType>::type, -1> eigenvalues(n);
  Eigen::Vector<typename ComplexDataType<TestType>::type, -1> new_eigenvalues(n);

  std::tie(eigenvector, eigenvalues) = CalculateGeneralEigenvalues<TestType>(A, false);
  nla_exam::DoubleShiftQrStep(A, false);

  std::tie(eigenvector, new_eigenvalues) =
      CalculateGeneralEigenvalues<TestType>(A, false);
  std::vector<typename ComplexDataType<TestType>::type> ev(eigenvalues.data(),
                                                           eigenvalues.data() + n);
  std::vector<typename ComplexDataType<TestType>::type> nev(new_eigenvalues.data(),
                                                            new_eigenvalues.data() + n);
  order_as_min_matching(ev, nev);
  for (unsigned int i = 0; i < ev.size(); ++i) {
    REQUIRE_THAT(std::abs(ev.at(i) - nev.at(i)),
                 Catch::Matchers::WithinAbs(0.0, tol<TestType>() * 1e2));
  }
}


TEMPLATE_TEST_CASE(
    "Double Shift QR Step with exceptional shift does not change the eigenvalues",
    "[DoubleShiftQRStep]", float, double) {
  int n = GENERATE(take(5, random<int>(3, 40)));
  Eigen::Matrix<TestType, -1, -1> A = Eigen::Matrix<TestType, -1, -1>::Random(n, n);
  nla_exam::HessenbergTransformation(A);

  Eigen::Matrix<TestType, -1, -1> eigenvector;
  Eigen::Vector<typename ComplexDataType<TestType>::type, -1> eigenvalues(n);
  Eigen::Vector<typename ComplexDataType<TestType>::type, -1> new_eigenvalues(n);

  std::tie(eigenvector, eigenvalues) = CalculateGeneralEigenvalues<TestType>(A, false);
  nla_exam::DoubleShiftQrStep(A, true);

  std::tie(eigenvector, new_eigenvalues) =
      CalculateGeneralEigenvalues<TestType>(A, false);
  std::vector<typename ComplexDataType<TestType>::type> ev(eigenvalues.data(),
                                                           eigenvalues.data() + n);
  std::vector<typename ComplexDataType<TestType>::type> nev(new_eigenvalues.data(),
                                                            new_eigenvalues.data() + n);
  order_as_min_matching(ev, nev);
  for (unsigned int i = 0; i < ev.size(); ++i) {
    REQUIRE_THAT(std::abs(ev.at(i) - nev.at(i)),
                 Catch::Matchers::WithinAbs(0.0, tol<TestType>() * 1e2));
  }
}
