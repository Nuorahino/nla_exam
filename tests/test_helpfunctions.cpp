#include <complex>
#include <filesystem>
#include <sstream>
#include <string>
#include <type_traits>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_random.hpp>
#include <eigen3/Eigen/Dense>

#include "../include/helpfunctions.hh"

class Test_class {};

inline bool
operator<(const Test_class &lhs, const Test_class &rhs) {
  return true;
}


// TODO (Georg): Is it correct, that no number types like "char" pass this check
// successfully?
TEMPLATE_TEST_CASE("Complex data types are correctly identified as complex",
                   "[is_complex]", std::complex<int>, std::complex<double>,
                   std::complex<unsigned int>, std::complex<char>, std::complex<bool>,
                   std::complex<Test_class>) {
  STATIC_CHECK(IsComplex<TestType>());
}


TEMPLATE_TEST_CASE("Non Complex data types are correctly identified as not complex",
                   "[is_complex]", int, double, unsigned int, char, std::string, bool,
                   Test_class) {
  STATIC_CHECK(!IsComplex<TestType>());
}


TEMPLATE_TEST_CASE("Complex data types are their own EvType", "[EvType]",
                   std::complex<int>, std::complex<double>, std::complex<unsigned int>,
                   std::complex<char>, std::complex<bool>) {
  STATIC_CHECK(
      std::is_same<TestType,
                   typename EvType<IsComplex<TestType>(), TestType>::type>::value);
}


TEMPLATE_TEST_CASE(
    "Non Complex data types get the complex specialization as their EvType", "[EvType]",
    int, double, unsigned int, char, bool) {
  STATIC_CHECK(
      std::is_same<std::complex<TestType>,
                   typename EvType<IsComplex<TestType>(), TestType>::type>::value);
}


TEMPLATE_TEST_CASE("Complex data types get correct DoubleType", "[DoubleType]",
                   std::complex<int>, std::complex<double>, std::complex<unsigned int>,
                   std::complex<char>, std::complex<bool>) {
  STATIC_CHECK(std::is_same<std::complex<double>,
                            typename DoubleType<IsComplex<TestType>()>::type>::value);
}


TEMPLATE_TEST_CASE("Non Complex data types get correct double type", "[DoubleType]", int,
                   double, unsigned int, char, bool) {
  STATIC_CHECK(
      std::is_same<double, typename DoubleType<IsComplex<TestType>()>::type>::value);
}


TEMPLATE_TEST_CASE("Correctly Identify Hermitian Matrices", "[IsHermitian]",
                   Eigen::MatrixXd, Eigen::MatrixXi, Eigen::MatrixXcd) {
  int n = GENERATE(10, 15);
  TestType Mat = TestType::Random(n, n);
  Mat = Mat + Mat.adjoint().eval();
  CHECK(IsHermitian(Mat));
}


TEMPLATE_TEST_CASE("Correctly Identify Non Hermitian Matrices", "[IsHermitian]",
                   Eigen::MatrixXd, Eigen::MatrixXi, Eigen::MatrixXcd) {
  int n = GENERATE(10, 15);
  TestType Mat = TestType::Random(n, n);
  CHECK(!IsHermitian(Mat));
}


TEMPLATE_TEST_CASE("Correctly Identify Non Square Matrices as not Hermitian",
                   "[IsHermitian]", Eigen::MatrixXd, Eigen::MatrixXi, Eigen::MatrixXcd) {
  int n = GENERATE(10, 15);
  int m = GENERATE(-5, -1, 1, 5);

  TestType Mat = TestType::Ones(n, n + m);
  CHECK(!IsHermitian(Mat));
}

TEMPLATE_TEST_CASE("Correctly Identify Datatypes with a '<' function", "[HasLesser]", int,
                   double, char, bool, std::string, Test_class) {
  CHECK(HasLesser<TestType>::check());
}

TEMPLATE_TEST_CASE("Correctly Identify Datatypes without a '<' function",
                   "[HasLesser]"
                   "[IsHermitian]",
                   Eigen::MatrixXd, std::complex<double>) {
  CHECK(!HasLesser<TestType>::check());
}


TEMPLATE_TEST_CASE("Lesser EV works for arithmetic data types", "[LesserEv]", int, double,
                   char, bool, unsigned int) {
  TestType a = 0;
  TestType b = 1;
  REQUIRE(LesserEv(a, b) == true);
  // TODO (Georg): Maybe use arithmetic here in place of double?
  if constexpr (std::is_same<TestType, double>()) {
    SECTION("LesserEV compare for random double values") {
      // TODO (Georg): Insert a random double generator
      double a = 1;
      double b = 2;
      REQUIRE(LesserEv(a, b) == (a < b));
    }
  }
}


TEMPLATE_TEST_CASE(
    "Lesser EV works for complex data types", "[LesserEv]",
    //     std::complex<int>, std::complex<double>, std::complex<unsigned int>) {
    std::complex<int>, std::complex<double>) {
  TestType a1(3, 2);
  TestType b1(3, 4);
  REQUIRE(LesserEv(a1, b1) == true);
  TestType a2(3, 4);
  TestType b2(3, 2);
  REQUIRE(LesserEv(a2, b2) == false);

  TestType a3(2, 3);
  TestType b3(4, 3);
  REQUIRE(LesserEv(a3, b3) == true);
  TestType a4(4, 3);
  TestType b4(2, 3);
  REQUIRE(LesserEv(a4, b4) == false);

  TestType a5(4, 3);
  TestType b5(4, 3);
  REQUIRE(LesserEv(a5, b5) == false);
}


TEST_CASE("Printing of Vectors works correctly", "[print std::vector]") {
  SECTION("Printing double Vector") {
    std::vector<double> Vec = {0.3, 2.11, 3};
    std::stringstream output;
    output << Vec;
    REQUIRE(output.str() == "{0.3, 2.11, 3}");
  }
  SECTION("Printing integer Vector") {
    std::vector<int> Vec_int = {0, 2, 3};
    std::stringstream output;
    output << Vec_int;
    REQUIRE(output.str() == "{0, 2, 3}");
  }
  SECTION("Printing string Vector") {
    std::vector<std::string> Vec_str = {"This", "is a", "test"};
    std::stringstream output;
    output << Vec_str;
    REQUIRE(output.str() == "{This, is a, test}");
  }
}


TEMPLATE_TEST_CASE("Converting Eigen Vectors to std::vector", "[Convert to Vector]", int,
                   float, double) {
  SECTION("Converting double vectors") {
    Eigen::Vector<TestType, 3> Vec{3, -5, 8};
    std::vector<TestType> res = {3, -5, 8};
    REQUIRE(res == ConvertToVec(Vec));
  }
}
