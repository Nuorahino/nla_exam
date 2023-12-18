#include <iostream>
#include <complex>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "qr.hh"
#include "test.hh"
#include "test_cases.hh"


// Maybe use std::optional instead
int main(int argc, char** argv) {
  std::cout << "Project Version: " << "Solution needed" << std::endl;
  Eigen::MatrixXd testMat{{2,3,4},{5,7,6},{8,9,10}};
  std::cout << testMat << std::endl;
  std::vector<double> x = nla_exam::GetGivensEntries<double>(2.0, 3.3);
  std::cout << x.at(0) << ", " << x.at(1) << std::endl;


  std::string filename = GetFileName();
  std::string basedir = "/home/georg/uni/11_sem23_24/ba/src/";

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
  //run_small_tests();

  std::cout << "start: " << start << std::endl;
  std::cout << "max_size: " << max_size << std::endl;
  for( int i = start; i <= max_size; ++i ) {
    RunTest<Eigen::MatrixXd>(summary_file, eigenvalue_file, i, seed, false, 1e-12);
    // This fails for i = 250
    RunTest<Eigen::MatrixXd>(summary_file, eigenvalue_file, i, seed, false, 1e-6);
#ifdef HALF
    RunTest<Eigen::MatrixXd>(summary_file, eigenvalue_file, i, seed, true, 1e-12);
    RunTest<Eigen::MatrixXcd>(summary_file, eigenvalue_file, i, seed, true, 1e-12);
    RunTest<Eigen::MatrixXcd>(summary_file, eigenvalue_file, i, seed, false, 1e-12);
#endif

#ifdef FULL
    RunTest<Eigen::MatrixXd>(summary_file, eigenvalue_file, i, seed, true, 1e-6);
    RunTest<Eigen::MatrixXcd>(summary_file, eigenvalue_file, i, seed, true, 1e-6);
    RunTest<Eigen::MatrixXcd>(summary_file, eigenvalue_file, i, seed, false, 1e-6);
#endif
  }

  return 1;
}
