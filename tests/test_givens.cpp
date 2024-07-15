#include <complex>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_random.hpp>
#include <eigen3/Eigen/Dense>

#include "../include/qr.hh"

TEMPLATE_TEST_CASE("Givens Parameter for first unit Vector is correct", "[givens]", int,
                   double, std::complex<int>, std::complex<double>) {
  TestType a = GENERATE(1, -2);
  TestType b = 0;
  TestType one = 1;

  auto res = nla_exam::GetGivensEntries(a, b);
  REQUIRE(res.at(1) == b);
  REQUIRE(res.at(0) == one);
}

TEMPLATE_TEST_CASE("Givens Parameter for first parameter = 0 is correct", "[givens]", int,
                   double, std::complex<int>, std::complex<double>) {
  TestType b = GENERATE(1, 0, -2);
  TestType a = 0;
  TestType one = 1;

  auto res = nla_exam::GetGivensEntries(a, b);
  REQUIRE(res.at(0) == a);
  REQUIRE(res.at(1) == one);
}
