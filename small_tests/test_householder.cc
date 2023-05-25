#include "test_householder.hh"

#include <iostream>


int main(int argc, char** argv) {
  Eigen::MatrixXcd testMat = Eigen::MatrixXd::Random(10,10);
  Eigen::MatrixXcd testMat1 = testMat;
  ApplyHouseholder(testMat(0, Eigen::lastN(9)), testMat1, 0, 1, 1e-8, std::polar(1.0, 1.5));
  Eigen::MatrixXcd testMat2 = testMat;
  ApplyHouseholder(testMat(0, Eigen::lastN(9)), testMat2, 0, 1, 1e-8, std::polar(2.0, 0.0));
  ApplyHouseholder(testMat(0, Eigen::lastN(9)), testMat, 0, 1, 1e-8);

  std::cout << "normal case" << std::endl;
  std::cout << testMat(0, Eigen::lastN(8)).norm() << std::endl;
  std::cout << "times Real" << std::endl;
  std::cout << testMat2(0, Eigen::lastN(8)).norm() << std::endl;
  std::cout << "times complex angle" << std::endl;
  std::cout << testMat1(0, Eigen::lastN(8)).norm() << std::endl;



  return 1;
}
