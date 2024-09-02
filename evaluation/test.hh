#ifndef TEST_HH_
#define TEST_HH_

#ifdef LAPACK
#define VERSION "LAPACK"
#else
#ifdef EIGEN
#define VERSION "EIGEN"
#else
#ifdef TWO_VEC
#define VERSION "elementwise2"
#else
#define VERSION "elementwise"
#endif
#endif
#endif

#include <fstream>
#include <vector>
#include <complex>
#include <string>
#include <chrono>

#include <eigen3/Eigen/Dense>

//#include "helpfunctions/helpfunctions.hh"
#include "helpfunctions.hh"
#include "qr.hh"
#include "createMatrix.hh"
#include "../tests/helpfunctions_for_test.hh"
#include "../lapack/lapack_interface_impl.hh"
#include "../matrix/matrix_classes.hh"

/*
 * Write the summary Header to the file
 * Parameter:
 * - a_file: File to write to
 * Return: void
 */
void PrintSummaryHeader(std::ofstream& a_file);

/*
 * Write the eigenvalue Header to the file
 * Parameter:
 * - a_file: File to write to
 * Return: void
 */
void PrintEigenvalueHeader(std::ofstream& a_file);

/*
 * Write the summary to the file
 * Parameter:
 * - a_file: File to write to
 * - ak_prefix: prefix string specifying the test options
 * - ak_error: vector of the estimation errors
 * Return: void
 */
void PrintSummary(std::ofstream& a_file, const std::string& ak_prefix,
    const std::vector<double>& ak_error);

/*
 * Write the eigenvalues to a the file
 * Parameter:
 * - a_file: File to write to
 * - ak_prefix: prefix string specifying the test options
 * - ak_estimate: vector of the estimations
 * - ak_error: vector of the estimation errors
 * Return: void
 */
void PrintEigenvalues(std::ofstream& a_file, const std::string& ak_prefix,
    const std::vector<std::complex<double>>& ak_estimate,
    const std::vector<double>& ak_error);

/*
 * Generate the filename to write test result to
 * Parameter:
 * Return: the filename
 */
std::string GetFileName();

/*
 * Generate the string defining the version that is tested
 * Parameter:
 * - ak_size: size of the problem
 * - ak_is_hermitian: bool
 * - ak_is_complex: bool
 * - ak_seed: seed used for matrix generation
 * - ak_tol: specify the tolerance
 * - ak_runtime: duration of the method
 * Return: summary of the information
 */
std::string GetVariantString(const int ak_size , const bool ak_is_hermitian,
    const bool ak_is_complex, const int ak_seed, const double ak_tol,
    const std::chrono::duration<double>& ak_runtime);

/*
 * Calculate the error of the estimate
 * Parameter:
 * - ak_estimate: vector of eigenvalue estimates
 * - ak_exact: vector of exact eigenvalues
 */
template<class DT>
std::vector<double> GetApproximationError(
    std::vector<DT>& ak_estimate,
    const std::vector<DT>& ak_exact ) {
  assert( ak_estimate.size() == ak_exact.size());
  order_as_min_matching(ak_estimate, ak_exact);
  std::vector<double> res;
  res.reserve(ak_estimate.size());
  for (unsigned i = 0; i < ak_estimate.size(); ++i) {
    res.push_back(std::abs(ak_estimate.at(i) - ak_exact.at(i)));
  }
  return res;
}

/*
 * Run a test for an eigenvalue estimator
 * Parameter:
 * - a_summary_file: file to write the summary to
 * - a_summary_file: file to write the estimate to
 * - ak_size: size of the matrix
 * - ak_seed: seed to initialize the matrix generation
 * - ak_hermitian: bool
 * - ak_tol: specify the tolerance
 */
template<class MatrixType>
void RunTest(std::ofstream& a_summary_file, [[maybe_unused]] std::ofstream& a_eigenvalue_file,
    const int ak_size, const int ak_seed, const bool ak_is_hermitian,
    const double ak_tol = 1e-12) {
  typedef std::complex<double> C;
  MatrixType M;
  std::vector<C> res;
  std::tie(M, res) = CreateRandom<MatrixType>(ak_size, ak_is_hermitian, ak_seed);

  nla_exam::HessenbergTransformation<>(M, ak_is_hermitian);

  auto start = std::chrono::steady_clock::now();
  std::vector<C> estimate;
#ifdef LAPACK
  tridiagonal_matrix2 t_mat{M.real()};
#else
#ifdef TWO_VEC
  tridiagonal_matrix2 t_mat{M.real()};
#else
  tridiagonal_matrix t_mat{M.real()};
#endif
#endif

#ifdef EIGEN
 Eigen::VectorXcd test;
#endif
  if (ak_is_hermitian) {
#ifdef LAPACK
  estimate = CalculateTridiagonalEigenvalues(t_mat.diag, t_mat.sdiag);
#else
#ifdef EIGEN
    Eigen::SelfAdjointEigenSolver<MatrixType> es(M, false);
    test = es.eigenvalues();
    if(es.info()) std::cout << "failed" << std::endl;
    estimate = ConvertToVec(test);
#else
    estimate = nla_exam::QrMethod<true>(t_mat, ak_tol);
#endif
#endif
  } else {
#ifdef EIGEN
    if constexpr (IsComplex<typename MatrixType::Scalar>()) {
      Eigen::ComplexEigenSolver<MatrixType> es(M, false);
      test = es.eigenvalues();
      if(es.info()) std::cout << "failed" << std::endl;
    } else {
      Eigen::EigenSolver<MatrixType> es(M, false);
      test = es.eigenvalues();
      if(es.info()) std::cout << "failed" << std::endl;
    }
    estimate = ConvertToVec(test);
#else
    //estimate = nla_exam::QrMethod<false>(M, ak_tol);
#endif
  }

  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> runtime = (end - start);

  std::sort(estimate.begin(), estimate.end(), [](C& a, C& b) {
      return LesserEv(a, b);});

  std::string prefix = GetVariantString(ak_size, ak_is_hermitian,
      IsComplex<typename MatrixType::Scalar>(), ak_seed, ak_tol, runtime);
  std::vector<double> error = GetApproximationError(estimate, res);
  PrintSummary(a_summary_file, prefix, error);
  //PrintEigenvalues(a_eigenvalue_file, prefix, estimate, error);

}


#endif
