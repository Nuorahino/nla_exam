#ifndef TEST_HH_
#define TEST_HH_

#include "sfinae.hh"
#include <string>

//#ifdef LAPACK
//#include "../lapack/lapack_interface_impl.hh"
//#endif
#ifdef BLAZE
#include <blaze/Math.h>
#endif
#ifdef ARMADILLO
#include <armadillo>
#endif

#include <fstream>
#include <vector>
#include <complex>
#include <string>
#include <chrono>

#include <eigen3/Eigen/Dense>

#include "helpfunctions.hh"
#include "qr.hh"
#include "createMatrix.hh"
#include "../tests/helpfunctions_for_test.hh"
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
std::string GetVariantString(const std::string ak_variant, const int ak_size , const bool ak_is_hermitian,
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


template<class MatrixType, class VectorType>
void RunTestNew(std::ofstream& a_summary_file, MatrixType Matrix,
    const VectorType res, const std::string ak_variant,
    const int ak_size, const int ak_seed, const bool ak_is_hermitian,
    const double ak_tol = 1e-12) {
  std::chrono::duration<double> runtime;
  VectorType estimate;
  if (ak_is_hermitian) {
    auto start = std::chrono::steady_clock::now();
    estimate = nla_exam::QrMethod<true>(Matrix, ak_tol);
    auto end = std::chrono::steady_clock::now();
    runtime = (end - start);
  } else {
    //estimate = nla_exam::QrMethod<false>(t_mat, ak_tol);
  }

  std::string prefix = GetVariantString(ak_variant, ak_size, ak_is_hermitian,
      IsComplex<typename ElementType<MatrixType>::type>(), ak_seed, ak_tol, runtime);
  std::vector<double> error = GetApproximationError(estimate, res);
  PrintSummary(a_summary_file, prefix, error);
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
  typedef typename ElementType<MatrixType>::type DataType;
  typedef typename EvType<IsComplex<DataType>(), DataType>::type  C;
  MatrixType M;
  std::vector<C> res;
  std::tie(M, res) = CreateRandom<MatrixType, C>(ak_size, ak_is_hermitian, ak_seed);

  nla_exam::HessenbergTransformation<>(M, ak_is_hermitian);

#ifdef TWO_VEC
  tridiagonal_matrix2<DataType> twovec_mat{M.real()};
  RunTestNew(a_summary_file, twovec_mat, res, "elementwise 2", ak_size, ak_seed, ak_is_hermitian, ak_tol);
#endif
#ifdef ONE_VEC
  tridiagonal_matrix<DataType> onevec_mat{M.real()};
  RunTestNew(a_summary_file, onevec_mat, res, "elementwise 1", ak_size, ak_seed, ak_is_hermitian, ak_tol);
#endif
#ifdef NESTED
  tridiagonal_matrix_nested<DataType> nested_mat{M.real()};
  RunTestNew(a_summary_file, nested_mat, res, "nested", ak_size, ak_seed, ak_is_hermitian, ak_tol);
#endif
#ifdef WRAPPED
  EigenWrapper<Eigen::Matrix<DataType, -1, -1>> wrapped_mat{M.real()};
  RunTestNew(a_summary_file, wrapped_mat, res, "wrapped", ak_size, ak_seed, ak_is_hermitian, ak_tol);
#endif
#ifdef BLAZE
  blaze::DynamicMatrix<DataType> blaze_mat(M.rows(), M.rows());
  for(int i = 0; i < M.rows(); ++i) {
    for(int j = 0; j < M.rows(); ++j) {
      blaze_mat(i, j) = M(i, j);
    }
  }
  RunTestNew(a_summary_file, blaze_mat, res, "blaze", ak_size, ak_seed, ak_is_hermitian, ak_tol);
#endif
#ifdef ARMADILLO
  arma::Mat<DataType> arma_mat(M.rows(), M.rows());
  for(int i = 0; i < M.rows(); ++i) {
    for(int j = 0; j < M.rows(); ++j) {
      arma_mat(i, j) = M(i, j);
    }
  }
  RunTestNew(a_summary_file, arma_mat, res, "armadillo", ak_size, ak_seed, ak_is_hermitian, ak_tol);
#endif
  std::vector<C> estimate;
#ifdef EIGEN
  {
    auto start = std::chrono::steady_clock::now();
    Eigen::SelfAdjointEigenSolver<MatrixType> es(M, false);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<DataType> runtime = (end - start);
    Eigen::Vector<C, -1> test = es.eigenvalues();
    if(es.info()) std::cout << "failed" << std::endl;
    estimate = ConvertToVec(test);

    std::string prefix = GetVariantString("Eigen", ak_size, ak_is_hermitian,
        IsComplex<typename ElementType<MatrixType>::type>(), ak_seed, ak_tol, runtime);
    std::vector<double> error = GetApproximationError(estimate, res);
    PrintSummary(a_summary_file, prefix, error);
  }
#endif
#ifdef LAPACK
  {
    tridiagonal_matrix2 temp_mat{M.real()};
    auto start = std::chrono::steady_clock::now();
    estimate = CalculateTridiagonalEigenvalues(temp_mat.diag, temp_mat.sdiag);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> runtime = (end - start);

    std::string prefix = GetVariantString("LAPACK", ak_size, ak_is_hermitian,
        IsComplex<typename ElementType<MatrixType>::type>(), ak_seed, ak_tol, runtime);
    std::vector<double> error = GetApproximationError(estimate, res);
    PrintSummary(a_summary_file, prefix, error);
  }
#endif
}
#endif
