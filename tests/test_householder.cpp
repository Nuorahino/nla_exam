#include <complex.h>

#include <catch2/catch_all.hpp>
#include <eigen3/Eigen/Dense>

#include "../include/qr.hh"
#include "../lapack/lapack_interface_impl.hh"
#include "helpfunctions_for_test.hh"

TEMPLATE_TEST_CASE("Householder Vector is unit vector, if the last n-1 entries are 0",
                   "[householder][compute_householder]", float, double,
                   std::complex<float>, std::complex<double>) {
  int n = GENERATE(take(5, random<int>(3, 10)));
  Eigen::Vector<TestType, -1> v = Eigen::Vector<TestType, -1>::Zero(n);
  v(0) = 1;
  Eigen::Vector<TestType, -1> w = nla_exam::GetHouseholderVector(v);
  REQUIRE_THAT(w(Eigen::seqN(1, n-1)).norm(), Catch::Matchers::WithinAbs(0.0, tol<TestType>()));
  REQUIRE_THAT(std::abs(w(0) - TestType{1.0}), Catch::Matchers::WithinAbs(0.0, tol<TestType>()));
}

TEMPLATE_TEST_CASE("Computed Householder Vector combined with LAPACK introduces zeros",
                   "[householder][compute_householder]", float, double,
                   std::complex<float>, std::complex<double>) {
  int n = GENERATE(take(5, random<int>(3, 10)));
  Eigen::Vector<TestType, -1> v = Eigen::Vector<TestType, -1>::Random(n);
  Eigen::Vector<TestType, -1> w_n = nla_exam::GetHouseholderVector(v);
  TestType beta = 2 / w_n.squaredNorm();
  Eigen::Vector<TestType, -1> res = apply_householder_left<TestType>(v, w_n, beta);
  REQUIRE_THAT(res(Eigen::lastN(n-1)).norm(), Catch::Matchers::WithinAbs(0.0, tol<TestType>()));
}

TEMPLATE_TEST_CASE("Computed Householder Vector combined with apply Householder left introduces zeros",
                   "[householder][compute_householder][apply_householder]", float, double,
                   std::complex<float>, std::complex<double>) {
  int n = GENERATE(take(5, random<int>(3, 10)));
  Eigen::Vector<TestType, -1> v = Eigen::Vector<TestType, -1>::Random(n);
  Eigen::Vector<TestType, -1> w = nla_exam::GetHouseholderVector(v);
  nla_exam::ApplyHouseholderLeft(w, v);
  REQUIRE_THAT(v(Eigen::lastN(n-1)).norm(), Catch::Matchers::WithinAbs(0.0, tol<TestType>()));
}

TEMPLATE_TEST_CASE("Computed Householder Vector combined with apply Householder right introduces zeros",
                   "[householder][compute_householder][apply_householder]", float, double,
                   std::complex<float>, std::complex<double>) {
  int n = GENERATE(take(5, random<int>(3, 10)));
  Eigen::Vector<TestType, -1> v = Eigen::Vector<TestType, -1>::Random(n);
  Eigen::Vector<TestType, -1> w = nla_exam::GetHouseholderVector(v);
  nla_exam::ApplyHouseholderRight(w.conjugate(), v.transpose());
  REQUIRE_THAT(v(Eigen::lastN(n-1)).norm(), Catch::Matchers::WithinAbs(0.0, tol<TestType>()));
}

TEMPLATE_TEST_CASE("LAPCAK Householder Vector with Apply left Householder introduces Zeros",
                   "[householder][compute_householder]", float, double,
                   std::complex<float>, std::complex<double>) {
  int n = GENERATE(take(5, random<int>(3, 10)));
  Eigen::Vector<TestType, -1> v = Eigen::Vector<TestType, -1>::Random(n);
  Eigen::Vector<TestType, -1> w;
  TestType tau;
  std::tie(w, tau) = get_householder<TestType>(v);
  Eigen::Vector<TestType, -1> wh = Eigen::Vector<TestType, -1>::Ones(n);
  wh(Eigen::lastN(n-1)) = w;

  nla_exam::ApplyHouseholderLeft(wh, v, complex_conj(tau));
  REQUIRE_THAT(v(Eigen::lastN(n-1)).norm(), Catch::Matchers::WithinAbs(0.0, tol<TestType>()));
}

TEMPLATE_TEST_CASE("LAPCAK Householder Vector with Apply right Householder introduces Zeros",
                   "[householder][compute_householder]", float, double,
                   std::complex<float>, std::complex<double>) {
  int n = GENERATE(take(5, random<int>(3, 10)));
  Eigen::Vector<TestType, -1> v = Eigen::Vector<TestType, -1>::Random(n);
  Eigen::Vector<TestType, -1> w;
  TestType tau;
  std::tie(w, tau) = get_householder<TestType>(v);
  Eigen::Vector<TestType, -1> wh = Eigen::Vector<TestType, -1>::Ones(n);
  wh(Eigen::lastN(n-1)) = w;

  nla_exam::ApplyHouseholderRight(wh.conjugate(), v.transpose(), complex_conj(tau));
  REQUIRE_THAT(v(Eigen::lastN(n-1)).norm(), Catch::Matchers::WithinAbs(0.0, tol<TestType>()));
}

TEMPLATE_TEST_CASE("Left Householder rotation does not change the vector norm",
                   "[householder][compute_householder][apply_householder]", float, double,
                   std::complex<float>, std::complex<double>) {
  int n = GENERATE(take(5, random<int>(3, 10)));
  Eigen::Vector<TestType, -1> v = Eigen::Vector<TestType, -1>::Random(n);
  Eigen::Vector<TestType, -1> v_copy = v;

  Eigen::Vector<TestType, -1> w = nla_exam::GetHouseholderVector(v);
  nla_exam::ApplyHouseholderLeft(w, v);
  REQUIRE_THAT(v.norm(), Catch::Matchers::WithinAbs(v_copy.norm(), tol<TestType>()));
}

TEMPLATE_TEST_CASE("Right Householder rotation does not change the vector norm",
                   "[householder][compute_householder][apply_householder]", float, double,
                   std::complex<float>, std::complex<double>) {
  int n = GENERATE(take(5, random<int>(3, 10)));
  Eigen::Vector<TestType, -1> v = Eigen::Vector<TestType, -1>::Random(n);
  Eigen::Vector<TestType, -1> v_copy = v;
  Eigen::Vector<TestType, -1> w = nla_exam::GetHouseholderVector(v);
  nla_exam::ApplyHouseholderRight(w.conjugate(), v.transpose());
  REQUIRE_THAT(v.norm(), Catch::Matchers::WithinAbs(v_copy.norm(), tol<TestType>()));
}

TEMPLATE_TEST_CASE("Left Householder rotation does not change the matrix norm",
                   "[householder][compute_householder][apply_householder]", float, double,
                   std::complex<float>, std::complex<double>) {
  int n = GENERATE(take(5, random<int>(3, 10)));
  Eigen::Vector<TestType, -1> v = Eigen::Vector<TestType, -1>::Random(n);
  Eigen::Matrix<TestType, -1, -1 > A = Eigen::Matrix<TestType, -1, -1>::Random(n, n);
  Eigen::Matrix<TestType, -1, -1 > A_copy = A;

  Eigen::Vector<TestType, -1> w = nla_exam::GetHouseholderVector(v);
  nla_exam::ApplyHouseholderLeft(w, v);
  REQUIRE_THAT(A.norm(), Catch::Matchers::WithinAbs(A_copy.norm(), tol<TestType>()));
}

TEMPLATE_TEST_CASE("Right Householder rotation does not change the matrix norm",
                   "[householder][compute_householder][apply_householder]", float, double,
                   std::complex<float>, std::complex<double>) {
  int n = GENERATE(take(5, random<int>(3, 10)));
  Eigen::Vector<TestType, -1> v = Eigen::Vector<TestType, -1>::Random(n);
  Eigen::Matrix<TestType, -1, -1 > A = Eigen::Matrix<TestType, -1, -1>::Random(n, n);
  Eigen::Matrix<TestType, -1, -1 > A_copy = A;

  Eigen::Vector<TestType, -1> w = nla_exam::GetHouseholderVector(v);
  nla_exam::ApplyHouseholderRight(w.conjugate(), A);
  REQUIRE_THAT(A.norm(), Catch::Matchers::WithinAbs(A_copy.norm(), tol<TestType>()));
}

TEMPLATE_TEST_CASE("Computed Householder Vector combined with apply Householder left introduces zeros in Matrix",
                   "[householder][compute_householder][apply_householder]", float, double,
                   std::complex<float>, std::complex<double>) {
  int constexpr n = 10;
  int i = GENERATE(take(5, random<int>(0, n-1)));
  Eigen::Matrix<TestType, -1, -1 > A = Eigen::Matrix<TestType, -1, -1>::Random(n, n);
  Eigen::Vector<TestType, -1> v = A(Eigen::all, i);
  Eigen::Vector<TestType, -1> w = nla_exam::GetHouseholderVector(v);
  nla_exam::ApplyHouseholderLeft(w, A);
  REQUIRE_THAT(A(Eigen::lastN(n-1), i).norm(), Catch::Matchers::WithinAbs(0.0, tol<TestType>()));
}

TEMPLATE_TEST_CASE("Computed Householder Vector combined with apply Householder right introduces zeros in Matrix",
                   "[householder][compute_householder][apply_householder]", float, double,
                   std::complex<float>, std::complex<double>) {
  int constexpr n = 10;
  int i = GENERATE(take(5, random<int>(0, n-1)));
  Eigen::Matrix<TestType, -1, -1 > A = Eigen::Matrix<TestType, -1, -1>::Random(n, n);
  Eigen::Vector<TestType, -1> v = A(i, Eigen::all);
  Eigen::Vector<TestType, -1> w = nla_exam::GetHouseholderVector(v);

  nla_exam::ApplyHouseholderRight(w.conjugate(), A);
  REQUIRE_THAT(A(i, Eigen::lastN(n-1)).norm(), Catch::Matchers::WithinAbs(0.0, tol<TestType>()));
}




//TEMPLATE_TEST_CASE("Computed Householder Vector is equal to LAPACK vector",
//                   "[householder][compute_householder]", float, double,
//                   std::complex<float>, std::complex<double>) {
//  int n = GENERATE(take(5, random<int>(3, 10)));
//  Eigen::Vector<TestType, -1> v = Eigen::Vector<TestType, -1>::Random(n);
//  Eigen::Vector<TestType, -1> w_n = nla_exam::GetHouseholderVector(v);
//  TestType w0 = w_n(0);
//  w_n /= w0;
//  Eigen::Vector<TestType, -1> w;
//  TestType tau;
//  std::tie(w, tau) = get_householder<TestType>(v);
//  std::cout << "w_n = " << w_n << std::endl;
//  std::cout << "w = " << w << std::endl;
//  REQUIRE_THAT((w_n(Eigen::lastN(n-1)) - w).norm(), Catch::Matchers::WithinAbs(0.0, tol<TestType>()));
//}
  //w_n = w_n / w_n(0);
  //TestType w0 = w_n(0);


  //w_n = w_n(Eigen::seqN(1, n-1)).eval();
//  Eigen::Vector<TestType, -1> w;
//  Eigen::Vector<TestType, -1> wh = Eigen::Vector<TestType, -1>::Ones(n);
//  TestType tau;
//  std::tie(w, tau) = get_householder<TestType>(v);
//  wh(Eigen::lastN(n-1)) = w;
//  Eigen::Matrix<TestType, -1, -1> res = apply_householder_left<TestType>(v, wh, tau);
  //std::cout << " v after = " << v << std::endl;

//TEMPLATE_TEST_CASE("Householder Vector computation is equal to LAPACK",
//                   "[householder][compute_householder]", std::complex<double>) {
//                   //"[householder][compute_householder]", float, double,
//                   //std::complex<float>, std::complex<double>) {
//  //int n = GENERATE(take(5, random<int>(3, 10)));
//  int n = 3;
//  Eigen::Matrix<TestType, -1, -1> A = Eigen::Matrix<TestType, -1, -1>::Random(n, n);
//  Eigen::Vector<TestType, -1> v = A(Eigen::all, 0);
//  std::cout << "before A = " << A << std::endl;
//  std::cout << "rows = " << v.rows() << ", cols = " << v.cols() << std::endl;
//  Eigen::Vector<TestType, -1> w;
//  TestType tau;
//  std::tie(w, tau) = get_householder<TestType>(v);
//  Eigen::Vector<TestType, -1> v2 = Eigen::Vector<TestType, -1>::Ones(n);
//  v2(Eigen::seqN(1, n-1)) = w;
//  //Eigen::Vector<TestType, -1> w = nla_exam::GetHouseholderVector(v);
//  std::cout << "right before v = " << v << std::endl;
//  nla_exam::ApplyHouseholderLeft(v2, A, tau);
//  //Eigen::Matrix<TestType, -1, -1> res = apply_householder_left<TestType>(A, w, tau);
//  //Eigen::Matrix<TestType, -1, -1> res = apply_householder_left<TestType>(v, w, tau);
//  //std::cout << "Result" << std::endl << res << std::endl;
//  std::cout << "Result" << std::endl << A << std::endl;
//  //REQUIRE_THAT(w(Eigen::seqN(1, n-1)).norm(), Catch::Matchers::WithinAbs(0.0, tol<TestType>()));
//}
