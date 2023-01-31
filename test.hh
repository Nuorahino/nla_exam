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

void printSummaryHeader(std::ofstream&);

void printEigenvalueHeader(std::ofstream&);

void printSummary(std::ofstream&, const std::string&,
    const std::vector<double>&);

void printEigenvalues(std::ofstream&, const std::string&,
    const std::vector<std::complex<double>>&, const std::vector<double>&);

std::string getFileName();

std::string getVariantString(const int, const bool, const bool,
    const int, const double, const std::chrono::duration<double>&);

std::vector<double> getApproximationError(
    const std::vector<std::complex<double>>,
    const std::vector<std::complex<double>> );

template<class Mat>
void runTest(std::ofstream& aFile, std::ofstream& aEvFile, const int aSize, const int aSeed,
    const bool aIsHermitian, const double aTol = 1e-14) {
  std::cout << "new test" << std::endl;
  typedef std::complex<double> C;
  Mat M;
  std::vector<C> res;
  std::tie(M,res) = CreateRandom<Mat>(aSize, aIsHermitian, aSeed);
  auto start = std::chrono::steady_clock::now();
  std::vector<C> estimate = nla_exam::qr_method(M, aTol);
  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> runtime = (end - start);
  std::sort(estimate.begin(), estimate.end(), [](C& a, C& b) { return lesser_ev(a, b);});
  std::string prefix = getVariantString(aSize, aIsHermitian, is_complex<typename Mat::Scalar>(), aSeed, aTol, runtime);
  std::vector<double> error = getApproximationError(estimate, res);
  printSummary(aFile, prefix, error);
  printEigenvalues(aEvFile, prefix, estimate, error);
//  Eigen::ComplexEigenSolver<Mat> es(M);
//  auto comp_eigenvalues = es.eigenvalues();
//  std::cout << "Results: Estimate, Eigen, exact, error" << std::endl;
//  std::cout << estimate << std::endl;
//  std::cout << comp_eigenvalues << std::endl;
//  std::cout << res << std::endl;
//  std::cout << error << std::endl;
}


#endif
