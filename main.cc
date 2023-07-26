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
  std::vector<std::complex<double>> estimate;
  Eigen::EigenSolver<Eigen::MatrixXd> es;
  Eigen::ComplexEigenSolver<Eigen::MatrixXcd> esc;
//
//  Eigen::MatrixXd test1 = doubleShiftMatrixProblem();
//  estimate = nla_exam::QrMethod<false>(test1);
//  std::cout << "test1: " << std::endl;
//  std::cout << estimate << std::endl << std::endl;
//  es.compute(test1);
//  std::cout << es.eigenvalues() << std::endl;
//  Eigen::MatrixXcd test2 = wilkinsonShiftMatrixProblem();
//  estimate = nla_exam::QrMethod<false>(test2);
//  std::cout << "test2: " << std::endl;
//  std::cout << estimate << std::endl << std::endl;;
//  esc.compute(test2);
//  std::cout << esc.eigenvalues() << std::endl;
//  Eigen::MatrixXcd test3 = Multiple0evProblem();
//  estimate = nla_exam::QrMethod<false>(test3);
//  std::cout << "test3: " << std::endl;
//  std::cout << estimate << std::endl << std::endl;
//  esc.compute(test3);
//  std::cout << esc.eigenvalues() << std::endl;
  //Eigen::MatrixXcd test4 = ZeroMatrixProblem();
//  Eigen::MatrixXd test4 = ZeroMatrixProblem();
//  estimate = nla_exam::QrMethod<true>(test4, 1e-30);
//  std::cout << "test4: " << std::endl;
//  std::cout << estimate << std::endl << std::endl;
//  es.compute(test4);
//  std::cout << es.eigenvalues() << std::endl;
//  std::cout << "start: " << start << std::endl;
//  std::cout << "max_size: " << max_size << std::endl;
  for( int i = start; i <= max_size; ++i ) {
    std::cout << "Test n = " << i << std::endl;
    RunTest<Eigen::MatrixXd>(summary_file, eigenvalue_file, i, seed, true, 1e-12);
    //RunTest<Eigen::MatrixXd>(summary_file, eigenvalue_file, i, seed, true, 1e-6);
   RunTest<Eigen::MatrixXd>(summary_file, eigenvalue_file, i, seed, false, 1e-12);
   RunTest<Eigen::MatrixXcd>(summary_file, eigenvalue_file, i, seed, true, 1e-12);
   RunTest<Eigen::MatrixXcd>(summary_file, eigenvalue_file, i, seed, false, 1e-12);
    //RunTest<Eigen::MatrixXcd>(summary_file, eigenvalue_file, i, seed, false, 1e-6);
  }

  return 1;
}
