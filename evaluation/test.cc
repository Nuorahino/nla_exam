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

#include "helpfunctions.hh"
#include "qr.hh"

void PrintSummaryHeader(std::ofstream& a_file) {
  a_file << "version, size, hermitian, complex, seed, tol, runtime in s,"
         << "min_error, max_error, avg_error" << std::endl;
}

void PrintEigenvalueHeader(std::ofstream& a_file) {
  a_file << "version, size, hermitian, complex, seed, tol, runtime in s,"
         << "index, eigenvalue, error" << std::endl;
}

void PrintSummary(std::ofstream& a_file, const std::string& ak_prefix,
    const std::vector<double>& ak_error) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnarrowing"
  a_file << ak_prefix << ","
         << *std::min_element(ak_error.begin(), ak_error.end()) << ","
         << *std::max_element(ak_error.begin(), ak_error.end()) << ","
         << std::accumulate(ak_error.begin(), ak_error.end(), 0.0)/
              double{ak_error.size()}
         << std::endl;
#pragma GCC diagnostic pop
}

void PrintEigenvalues(std::ofstream& a_file, const std::string& ak_prefix,
                      const std::vector<std::complex<double>>& ak_estimate,
                      const std::vector<double>& ak_error) {
  assert(ak_estimate.size() == ak_error.size());
  for (long unsigned int i = 0; i < ak_estimate.size(); ++i) {
    a_file << ak_prefix << ","
           << i << ","
           << std::setprecision(std::numeric_limits<double>::max_digits10)
           << ak_estimate.at(i).real() << "+"
           << ak_estimate.at(i).imag() << "i,"
           << ak_error.at(i) << std::endl;
  }
}

std::string GetFileName() {
  std::stringstream res;
  auto now = std::chrono::system_clock::now();
  auto time = std::chrono::system_clock::to_time_t(now);
  res << std::put_time(std::localtime(&time), "%Y-%m-%d %X");
  return res.str();
}


std::string GetVariantString(const int ak_size, const bool ak_is_hermitian,
    const bool ak_is_complex, const int ak_seed, const double ak_tol,
    const std::chrono::duration<double>& ak_runtime) {
  std::stringstream res;
  res << VERSION << "," // Try to a import the Version from Cmake
      << ak_size << ","
      << ak_is_hermitian << ","
      << ak_is_complex << ","
      << ak_seed << ","
      << ak_tol << ","
      << ak_runtime.count();

  return res.str();
}


//std::vector<double> GetApproximationError(
//    std::vector<std::complex<double>>& ak_estimate,
//    const std::vector<std::complex<double>>& ak_exact) {
//  assert( ak_estimate.size() == ak_exact.size());
//  order_as_min_matching(ak_estimate, ak_exact);
//  std::vector<double> res;
//  res.reserve(ak_estimate.size());
//  for (unsigned i = 0; i < ak_estimate.size(); ++i) {
//    res.push_back(std::abs(ak_estimate.at(i) - ak_exact.at(i)));
//  }
//  return res;
//}
