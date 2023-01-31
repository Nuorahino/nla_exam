#include <iostream>
#include <complex>

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>

#include "qr.hh"
#include "test.hh"

// Maybe use std::optional instead
int main([[maybe_unused]] int argc, [[maybe_unused]] char** argv) {
  std::string filename = getFileName();

  std::ofstream summaryFile("../testresults/summary/" + filename);
  std::ofstream EigenvalueFile("../testresults/eigenvalues/" + filename);
  printSummaryHeader(summaryFile);
  printEigenvalueHeader(EigenvalueFile);

  int n = 50;
  int seed = 28;
  for( int i = 1; i < n; ++i ) {
//    runTest<Eigen::MatrixXd>(summaryFile, EigenvalueFile, i, seed, true);
//    runTest<Eigen::MatrixXd>(summaryFile, EigenvalueFile, i, seed, false);
    runTest<Eigen::MatrixXcd>(summaryFile, EigenvalueFile, i, seed, true);
//    runTest<Eigen::MatrixXcd>(summaryFile, EigenvalueFile, i, seed, false);
  }

  return 1;
}
