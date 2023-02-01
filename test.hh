#ifndef TEST_HH_
#define TEST_HH_

#include <fstream>
#include <vector>
#include <complex>
#include <string>
#include <chrono>

#include <eigen3/Eigen/Dense>

#include "helpfunctions/helpfunctions.hh"
#include "qr.hh"

void PrintSummaryHeader(std::ofstream& a_file);

void PrintEigenvalueHeader(std::ofstream& a_file);

void PrintSummary(std::ofstream& a_file, const std::string& ak_prefix,
    const std::vector<double>& ak_error);

void PrintEigenvalues(std::ofstream& a_file, const std::string& ak_prefix,
    const std::vector<std::complex<double>>& ak_estimate,
    const std::vector<double>& ak_error);

std::string GetFileName();

std::string GetVariantString(const int ak_size , const bool ak_is_hermitian,
    const bool ak_is_complex, const int ak_seed, const double ak_tol,
    const std::chrono::duration<double>& ak_runtime);

std::vector<double> GetApproximationError(
    const std::vector<std::complex<double>>& ak_estimate,
    const std::vector<std::complex<double>>& ak_exact );

template<class MatrixType>
void RunTest(std::ofstream& a_summary_file, std::ofstream& a_eigenvalue_file,
    const int ak_size, const int ak_seed, const bool ak_is_hermitian,
    const double ak_tol = 1e-14) {
  std::cout << "new test" << std::endl;
  typedef std::complex<double> C;
  MatrixType M;
  std::vector<C> res;
  std::tie(M,res) = CreateRandom<MatrixType>(ak_size, ak_is_hermitian, ak_seed);
  auto start = std::chrono::steady_clock::now();
  std::vector<C> estimate = nla_exam::QrMethod(M, ak_tol);
  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> runtime = (end - start);
  std::sort(estimate.begin(), estimate.end(), [](C& a, C& b) {
      return LesserEv(a, b);});
  std::string prefix = GetVariantString(ak_size, ak_is_hermitian,
      IsComplex<typename MatrixType::Scalar>(), ak_seed, ak_tol, runtime);
  std::vector<double> error = GetApproximationError(estimate, res);
  PrintSummary(a_summary_file, prefix, error);
  PrintEigenvalues(a_eigenvalue_file, prefix, estimate, error);

//  Eigen::ComplexEigenSolver<MatrixType> es(M);
//  auto comp_eigenvalues = es.eigenvalues();
//  std::cout << "Results: Estimate, Eigen, exact, error" << std::endl;
//  std::cout << estimate << std::endl;
//  std::cout << comp_eigenvalues << std::endl;
//  std::cout << res << std::endl;
//  std::cout << error << std::endl;
}


#endif
