#include <iostream>
#include <complex>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "qr.hh"
#include "test.hh"


// Maybe use std::optional instead
int main(int argc, char** argv) {
  //std::cout << "Project Version: " << VERSION << std::endl;
  std::cout << "Start Various Versions: " << std::endl;

  std::string filename = GetFileName();
  std::string basedir = "/home/georg/uni/12_sem24/ba2/";

  std::ofstream summary_file(basedir + "testresults/summary/" + filename);
  //std::ofstream eigenvalue_file(basedir + "testresults/eigenvalues/" + filename);
  std::ofstream eigenvalue_file;
  PrintSummaryHeader(summary_file);
  //PrintEigenvalueHeader(eigenvalue_file);

  int max_size = 1000;
  int start = 1;
  int seed = 28;
  double tol = 1e-12;
  if(argc == 4) {
    start = atoi(argv[1]);
    max_size = atoi(argv[2]);
    seed = atoi(argv[3]);
  }
  if(argc == 5) {
    start = atoi(argv[1]);
    max_size = atoi(argv[2]);
    seed = atoi(argv[3]);
    tol = std::stod(argv[4]);
  }
  std::cout << tol << std::endl;


  std::cout << "Testing for: " << start << " to " << max_size << std::endl;
  for( int i = start; i <= max_size; ++i ) {

#ifdef REAL_SYMM
    RunTest<Eigen::MatrixXd>(summary_file, eigenvalue_file, i, seed, true, tol);
#endif
#ifdef LONGREAL_SYMM
    RunTest<Eigen::Matrix<long double, -1, -1>>(summary_file, eigenvalue_file, i, seed, true, tol);
#endif
#ifdef FLOAT_SYMM
    RunTest<Eigen::MatrixXf>(summary_file, eigenvalue_file, i, seed, true, tol);
#endif

#ifdef REAL_NON_SYMM
    RunTest<Eigen::MatrixXd>(summary_file, eigenvalue_file, i, seed, false, tol);
#endif
#ifdef COMPLEX_SYMM
    RunTest<Eigen::MatrixXcd>(summary_file, eigenvalue_file, i, seed, true, tol);
#endif
#ifdef COMPLEX_NON_SYMM
    RunTest<Eigen::MatrixXcd>(summary_file, eigenvalue_file, i, seed, false, tol);
#endif
  }
  return 1;
}
