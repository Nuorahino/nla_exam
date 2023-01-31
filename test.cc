#include "test.hh"

#include <cassert>
#include <algorithm>
#include <numeric>
#include <complex>
#include <sstream>
#include <ctime>
#include <iomanip>
#include <limits>

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>

#include "helpfunctions/helpfunctions.hh"
#include "qr.hh"

void printSummaryHeader(std::ofstream& aFile) {
  aFile << "version, size, hermitian, complex, seed, tol, runtime in s, min_error, max_error, avg_error" << std::endl;
}

void printEigenvalueHeader(std::ofstream& aFile) {
  aFile << "version, size, hermitian, complex, seed, tol, runtime in s, index, eigenvalue, error" << std::endl;
}

void printSummary(std::ofstream& aFile, const std::string& aPrefix,
    const std::vector<double>& aError) {
  aFile << aPrefix << ","
        << *std::min_element(aError.begin(), aError.end()) << ","
        << *std::max_element(aError.begin(), aError.end()) << ","
        << std::accumulate(aError.begin(), aError.end(), 0.0)/double(aError.size())
        << std::endl;
}

void printEigenvalues(std::ofstream& aFile, const std::string& aPrefix,
                            const std::vector<std::complex<double>>& aEstimate,
                            const std::vector<double>& aError) {
  assert(aEstimate.size() == aError.size());
  for(long unsigned int i = 0; i < aEstimate.size(); ++i) {
    aFile << aPrefix << ","
          << i << ","
          << std::setprecision(std::numeric_limits<double>::max_digits10)
          << aEstimate.at(i).real() << "+" << aEstimate.at(i).imag() << "i,"
          << aError.at(i) << std::endl;
  }
}

std::string getFileName() {
  std::stringstream res;
  auto now = std::chrono::system_clock::now();
  auto time = std::chrono::system_clock::to_time_t(now);
  res << std::put_time(std::localtime(&time), "%Y-%m-%d %X");
  return res.str();
}


std::string getVariantString(const int aSize, const bool aIsHermitian,
    const bool aIsComplex, const int aSeed, const double aTol,
    const std::chrono::duration<double>& aRuntime) {
  std::stringstream res;
  res << VERSION << ","
      << aSize << ","
      << aIsHermitian << ","
      << aIsComplex << ","
      << aSeed << ","
      << aTol << ","
      << aRuntime.count();

  return res.str();
}


std::vector<double> getApproximationError(
    const std::vector<std::complex<double>> aEstimate,
    const std::vector<std::complex<double>> aExact) {
  assert( aEstimate.size() == aExact.size());
  std::vector<double> res;
  res.reserve(aEstimate.size());
  for( auto estimate : aEstimate) {
    double min = std::numeric_limits<double>::max();
    for( auto exact : aExact) {
      double dist = std::abs(estimate - exact);
      if(dist < min) min = dist;
    }
    res.push_back(min);
  }
  return res;
}
