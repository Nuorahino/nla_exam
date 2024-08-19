#include <catch2/catch_all.hpp>

#include <algorithm>

#include "helpfunctions_for_test.hh"

TEMPLATE_TEST_CASE("Min Matching test", "[test]", float, double) {
  std::vector<TestType> a = {3.0, 1.0, 2.0};
  std::vector<TestType> b = {0.1, 1.1, 2.1};

  min_matching(a,b);
  for ( int i = 0; i < 3; ++i) {
    REQUIRE_THAT(a.at(i) - b.at(i), Catch::Matchers::WithinAbs(0.9, tol<TestType>()));
  }
}



TEMPLATE_TEST_CASE("Min Matching test", "[test]", std::complex<float>, std::complex<double>) {
  int n = GENERATE(take(5, random<int>(3, 20)));
  std::vector<TestType> a;
  for(int i = 0; i < n; ++i) {
    a.push_back(GENERATE(take(1,ComplexRandom<TestType>(-100.0, 100.0))));
  }
  std::vector<TestType> b = a;
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(b.begin(), b.end(), g);

  min_matching(a,b);
  for ( int i = 0; i < 3; ++i) {
    REQUIRE_THAT(std::abs(a.at(i) - b.at(i)), Catch::Matchers::WithinAbs(0, tol<TestType>()));
  }
}
