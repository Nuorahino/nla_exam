#include <iostream>
#include <complex>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <easy/profiler.h>

#include "qr.hh"
#include "test.hh"


// Maybe use std::optional instead
int main(int argc, char** argv) {
  EASY_PROFILER_ENABLE;
  std::string version = "new";
#ifdef EIGEN
  version = "EIGEN";
#endif
  std::cout << "Project Version: " << version << std::endl;

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


//  std::vector<int> testsizes;
//  testsizes = {10, 50, 100, 1000};
  profiler::startListen();

  //std::cout << "Testing for: " << testsizes << std::endl;
  std::cout << "Testing for: " << start << " to " << max_size << std::endl;
  for( int i = start; i <= max_size; ++i ) {
    std::string cur_size = std::to_string(i);
  //for (int i : testsizes) {
#ifdef FULL
#ifdef REAL_SYMM
  profiler::startListen();
    RunTest<Eigen::MatrixXd>(summary_file, eigenvalue_file, i, seed, true, 1e-12);
    profiler::dumpBlocksToFile((basedir + "testresults/profiling/real_symm/" + filename + "size" + cur_size + ".prof").c_str());
#endif
#ifdef REAL_NON_SYMM
    RunTest<Eigen::MatrixXd>(summary_file, eigenvalue_file, i, seed, false, 1e-12);
    profiler::dumpBlocksToFile((basedir + "testresults/profiling/real_non_symm/" + filename + "size" + cur_size + ".prof").c_str());
#endif
#ifdef COMPLEX_SYMM
    RunTest<Eigen::MatrixXcd>(summary_file, eigenvalue_file, i, seed, true, 1e-12);
    profiler::dumpBlocksToFile((basedir + "testresults/profiling/complex_symm/" + filename + "size" + cur_size + ".prof").c_str());
#endif
#ifdef COMPLEX_NON_SYMM
    RunTest<Eigen::MatrixXcd>(summary_file, eigenvalue_file, i, seed, false, 1e-12);
    profiler::dumpBlocksToFile((basedir + "testresults/profiling/complex_non_symm/" + filename + "size" + cur_size + ".prof").c_str());
#endif
#endif

#ifdef HALF
#ifdef REAL_SYMM
  profiler::startListen();
    RunTest<Eigen::MatrixXd>(summary_file, eigenvalue_file, i, seed, true, 1e-8);
    profiler::dumpBlocksToFile((basedir + "testresults/profiling/real_symm_half/" + filename + "size" + cur_size + ".prof").c_str());
#endif
#ifdef REAL_NON_SYMM
    RunTest<Eigen::MatrixXd>(summary_file, eigenvalue_file, i, seed, false, 1e-8);
    profiler::dumpBlocksToFile((basedir + "testresults/profiling/real_non_symm_half/" + filename + "size" + cur_size + ".prof").c_str());
#endif
#ifdef COMPLEX_SYMM
    RunTest<Eigen::MatrixXcd>(summary_file, eigenvalue_file, i, seed, true, 1e-8);
    profiler::dumpBlocksToFile((basedir + "testresults/profiling/complex_symm_half/" + filename + "size" + cur_size + ".prof").c_str());
#endif
#ifdef COMPLEX_NON_SYMM
    RunTest<Eigen::MatrixXcd>(summary_file, eigenvalue_file, i, seed, false, 1e-8);
    profiler::dumpBlocksToFile((basedir + "testresults/profiling/complex_non_symm_half/" + filename + "size" + cur_size + ".prof").c_str());
#endif

#endif
  }
  return 1;
}
