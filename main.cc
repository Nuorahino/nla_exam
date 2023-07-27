#include <iostream>
#include <complex>

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>

#include "qr.hh"
#include "test.hh"
#include "test_cases.hh"

// Maybe use std::optional instead
int main(int argc, char** argv) {
  std::string filename = GetFileName();
  std::string basedir = "/home/georg/uni/10_sem23/ba/src/nla_exam/";

  std::ofstream summary_file(basedir + "testresults/summary/" + filename);
  std::ofstream eigenvalue_file(basedir + "testresults/eigenvalues/" + filename);
  PrintSummaryHeader(summary_file);
  PrintEigenvalueHeader(eigenvalue_file);

  int max_size = 1000;
  int start = 1;
  int seed = 28;
  if(argc == 4) {
    start = atoi(argv[1]);
    max_size = atoi(argv[2]);
    seed = atoi(argv[3]);
  }
//  run_small_tests();

  std::cout << "start: " << start << std::endl;
  std::cout << "max_size: " << max_size << std::endl;
  for( int i = start; i <= max_size; ++i ) {
    std::cout << "Test n = " << i << std::endl;
    RunTest<Eigen::MatrixXd>(summary_file, eigenvalue_file, i, seed, true, 1e-12);
    RunTest<Eigen::MatrixXd>(summary_file, eigenvalue_file, i, seed, false, 1e-12);
    RunTest<Eigen::MatrixXcd>(summary_file, eigenvalue_file, i, seed, true, 1e-12);
    RunTest<Eigen::MatrixXcd>(summary_file, eigenvalue_file, i, seed, false, 1e-12);

    RunTest<Eigen::MatrixXd>(summary_file, eigenvalue_file, i, seed, true, 1e-6);
    RunTest<Eigen::MatrixXd>(summary_file, eigenvalue_file, i, seed, false, 1e-6);
    RunTest<Eigen::MatrixXcd>(summary_file, eigenvalue_file, i, seed, true, 1e-6);
    RunTest<Eigen::MatrixXcd>(summary_file, eigenvalue_file, i, seed, false, 1e-6);
  }

  return 1;
}
