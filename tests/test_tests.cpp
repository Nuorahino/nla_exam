#include <catch2/catch_all.hpp>

#include <algorithm>

#include "helpfunctions_for_test.hh"

TEMPLATE_TEST_CASE("Min Matching test", "[test]", float, double) {
  std::vector<TestType> a = {3.0, 1.0, 2.0};
  std::vector<TestType> b = {0.1, 1.1, 2.1};

  order_as_min_matching(a, b);
  for (int i = 0; i < 3; ++i) {
    REQUIRE_THAT(a.at(i) - b.at(i), Catch::Matchers::WithinAbs(0.9, tol<TestType>()));
  }
}


TEMPLATE_TEST_CASE("Min Matching already matched", "[test]", float, double) {
  std::vector<TestType> a = {3.0, 1.0, 2.0};
  std::vector<TestType> b = {3.0, 1.0, 2.0};

  order_as_min_matching(a, b);
  for (int i = 0; i < 3; ++i) {
    REQUIRE_THAT(a.at(i) - b.at(i), Catch::Matchers::WithinAbs(0.0, tol<TestType>()));
  }
}


TEMPLATE_TEST_CASE("Min Matching test", "[test]", std::complex<float>,
                   std::complex<double>) {
  int n = GENERATE(take(5, random<int>(3, 20)));
  std::vector<TestType> a;
  for (int i = 0; i < n/2; ++i) {
    a.push_back(GENERATE(take(1, ComplexRandom<TestType>(-100.0, 100.0))));
    a.push_back(GENERATE(take(1, ComplexRandom<TestType>(-100.0, 100.0))));
  }
  std::vector<TestType> b = a;
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(b.begin(), b.end(), g);

  std::cout << "a: " << a << std::endl;
  std::cout << "b: " << a << std::endl;
  order_as_min_matching(a, b);
  for (int i = 0; i < 3; ++i) {
    REQUIRE_THAT(std::abs(a.at(i) - b.at(i)),
                 Catch::Matchers::WithinAbs(0, tol<TestType>()));
  }
}
