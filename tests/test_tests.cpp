
#include <iostream>

#include <catch2/catch_all.hpp>

#include "helpfunctions_for_test.hh"

TEMPLATE_TEST_CASE("Min Matching test", "[test]", float, double) {
  std::vector<TestType> a = {3.0, 1.0, 2.0};
  //std::vector<TestType> a = {2.0, 1.0, 3.0};
  std::vector<TestType> b = {0.1, 1.1, 2.1};

  min_matching(a,b);
  for ( int i = 0; i < 3; ++i) {
    REQUIRE_THAT(a.at(i) - b.at(i), Catch::Matchers::WithinAbs(0.9, tol<TestType>()));
  }
}
