#include <iostream>
#include <complex>

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>

#include "qr.hh"
#include "test.hh"

// Maybe use std::optional instead
int main([[maybe_unused]] int argc, [[maybe_unused]] char** argv) {
  std::string filename = GetFileName();

  std::ofstream summary_file("../testresults/summary/" + filename);
  std::ofstream eigenvalue_file("../testresults/eigenvalues/" + filename);
  PrintSummaryHeader(summary_file);
  PrintEigenvalueHeader(eigenvalue_file);

  int max_size = 50;
  int seed = 28;
  for( int i = 1; i < max_size; ++i ) {
    //RunTest<Eigen::MatrixXd>(summary_file, eigenvalue_file, i, seed, true);
    RunTest<Eigen::MatrixXd>(summary_file, eigenvalue_file, i, seed, false, 1e-6);
    //RunTest<Eigen::MatrixXcd>(summary_file, eigenvalue_file, i, seed, true);
    //RunTest<Eigen::MatrixXcd>(summary_file, eigenvalue_file, i, seed, false);
  }

  return 1;
}
