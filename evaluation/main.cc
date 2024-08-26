#include <iostream>
#include <complex>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <easy/profiler.h>

#include "qr.hh"
#include "test.hh"
#include "test_cases.hh"


// Maybe use std::optional instead
int main(int argc, char** argv) {
  EASY_PROFILER_ENABLE;
  std::cout << "Project Version: " << "Solution needed" << std::endl;

  std::string filename = GetFileName();
  std::string basedir = "/home/georg/uni/12_sem24/ba2/";

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


  std::vector<int> testsizes;
  testsizes = {10, 50, 100, 1000};
  profiler::startListen();

  std::cout << "Testing for: " << testsizes << std::endl;
  //for( int i = start; i <= max_size; ++i ) {
  for (int i : testsizes) {
  profiler::startListen();
    RunTest<Eigen::MatrixXd>(summary_file, eigenvalue_file, i, seed, true, 1e-12);
  profiler::dumpBlocksToFile("test_profile1.prof");
    RunTest<Eigen::MatrixXd>(summary_file, eigenvalue_file, i, seed, true, 1e-6);
  profiler::dumpBlocksToFile("test_profile2.prof");
//  profiler::startListen();
//    RunTest<Eigen::MatrixXd>(summary_file, eigenvalue_file, i, seed, false, 1e-12);
//  profiler::dumpBlocksToFile("test_profile1.prof");
//    RunTest<Eigen::MatrixXd>(summary_file, eigenvalue_file, i, seed, false, 1e-6);
//  profiler::dumpBlocksToFile("test_profile2.prof");
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
  profiler::dumpBlocksToFile("test_profile.prof");
  return 1;
}
