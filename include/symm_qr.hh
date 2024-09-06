#ifndef SYMM_QR_HH_
#define SYMM_QR_HH_

#include <type_traits>
#include <cmath> // for std::copysign
#include <vector>

#include <easy/profiler.h>

#include "helpfunctions.hh"


namespace nla_exam {

template <class DataType, bool is_symmetric, class Matrix>
inline std::enable_if_t<is_symmetric, std::vector<DataType>>
CalcEigenvaluesFromSchur(const Matrix &ak_matrix,
                         [[maybe_unused]] const bool ak_matrix_is_diagonal = false) {
  std::vector<DataType> res(ak_matrix.rows());
  for (unsigned i = 0; i < ak_matrix.rows(); ++i) {
    res.at(i) = ak_matrix(i, i);
  }
  return res;
}


/* Get Wilkinson shift parameter for a given 2x2 Matrix
 * Parameter:
 * - ak_matrix: 2x2 Matrix of which to calculate the shift parameter
 * Return: value of the shift parameter
 */
template <class DataType, bool is_symmetric, class Matrix>
inline std::enable_if_t<is_symmetric && std::is_arithmetic<DataType>::value, DataType>
WilkinsonShift(const Matrix &ak_matrix, const int i) {
  //EASY_FUNCTION(profiler::colors::Red);
  DataType d = (ak_matrix(i, i) - ak_matrix(i + 1, i + 1)) / static_cast<DataType>(2.0);
  if (d >= 0) {
    return ak_matrix(i + 1, i + 1) + d - std::hypot(d, ak_matrix(i + 1, i));
  } else {
    return ak_matrix(i + 1, i + 1) + d + std::hypot(d, ak_matrix(i + 1, i));
  }
}


/* Deflates a Matrix converging to a diagonal matrix
 * Parameter:
 * - a_matrix: Matrix to deflate
 * - a_begin: Index of fhe block that is solved currently
 * - a_end: Index of fhe End that is solved currently
 * - ak_tol: Tolerance for considering a value 0
 * Return: "true" if the block is fully solved, "false" otherwise
 */
template <class Matrix>
inline int
DeflateDiagonal(Matrix &a_matrix, int &a_begin, int &a_end,
                const double ak_tol = 1e-12) {
  int state = 2;
  for (int i = a_end; i > a_begin; --i) {
    if (std::abs(a_matrix(i, i - 1)) <=
        (ak_tol * (std::abs(a_matrix(i, i)) + std::abs(a_matrix(i - 1, i - 1))))) {
      a_matrix(i, i - 1) = 0;
      if (state < 2) {
        a_begin = i;
        return 1;  // Subblock to solve found
      }
    } else if (state == 2) {  // Start of the block found
      if (i == a_end) {
        state = 0;
      } else {
        a_end = i;
        state = 1;
      }
    }
  }
  return state;
}


/* Calculate the entries of a Givens Matrix
 * Parameter:
 * - ak_a: first entry
 * - ak_b: entry to eliminate
 * Return: Vector containing {c, s}
 */
template <class DataType>
inline std::enable_if_t<std::is_arithmetic<DataType>::value, std::vector<DataType>>
GetGivensEntries(const DataType &ak_a, const DataType &ak_b) {
  //EASY_FUNCTION(profiler::colors::Red);
  std::vector<DataType> res(3);
  if (std::abs(ak_a) <= std::numeric_limits<DataType>::epsilon()) {
    res.at(0) = 0;
    res.at(1) = 1;
  } else {
    DataType r = std::hypot(ak_a, ak_b);
    res.at(0) = std::abs(ak_a) / r;
    res.at(1) = ak_b / r * DataType{std::copysign( DataType{1}, ak_a)};
    res.at(2) = res.at(1);
    // TODO(Georg): instead of copysign maybe use a test with >
                                  // 0 to do this, as this always converts to float
  }
  return res;
}


template <bool first, bool last, bool is_symmetric,
  class DataType, class Matrix>
inline std::enable_if_t<std::is_arithmetic<DataType>::value && is_symmetric, void>
ApplyGivensTransformation(Matrix &matrix, const DataType ak_c,
                const DataType ak_s, const int aBegin, DataType &buldge) {
  //EASY_FUNCTION(profiler::colors::Red);
    int k = aBegin;
    DataType c = ak_c;
    DataType s = ak_s;
    DataType x1 = c * c * matrix(k, k) + s * s * matrix(k+1,k+1) + 2 * c * s * matrix(k, k+1);
    DataType a2 = c * -s * matrix(k,k) - s * s * matrix(k, k+1) + c * c * matrix(k, k+1) + s * c * matrix(k+1, k+1);
    DataType x2 = c * c * matrix(k+1, k+1) + s * s * matrix(k,k) - 2 * c * s * matrix(k+1, k);

if constexpr (!first) {
    DataType a1 = c * matrix(k, k - 1) + s * buldge;
    matrix(k-1, k) = a1;
    matrix(k, k-1) = a1;
}

if constexpr (!last) {
    buldge = s * matrix(k+2, k+1);
    DataType a3 = c * matrix(k+1, k+2);
    matrix(k+1, k+2) = a3;
    matrix(k+2, k+1) = a3;
}

    matrix(k+1, k) = a2;
    matrix(k, k+1) = a2;
    matrix(k, k) = x1;
    matrix(k+1, k+1) = x2;

  return;
}


/* Executes one step of the implicit qr algorithm for a tridiagonal Matrix
 * Parameter:
 * - a_matrix: Tridiagonal Matrix
 * Return: void
 */
template <class DataType, bool is_symmetric, typename Matrix>
std::enable_if_t<std::is_arithmetic<DataType>::value && is_symmetric, void>
ImplicitQrStep(Matrix &matrix,
               const int aBegin, const int aEnd) {
  //EASY_FUNCTION(profiler::colors::Red);
  int n = aEnd - aBegin + 1;
  DataType shift = WilkinsonShift<DataType, is_symmetric>(
        matrix, aEnd - 1);
  auto entries = GetGivensEntries<>(matrix(aBegin, aBegin) - shift,
                                    matrix(aBegin + 1, aBegin));  // Parameter for the initial step
  // innitial step
  DataType buldge = 0;
  if (n == 2) { //
    ApplyGivensTransformation<true, true, true, DataType>(matrix, entries.at(0), entries.at(1), aBegin, buldge);
    return;
  }
  ApplyGivensTransformation<true, false, true, DataType>(matrix, entries.at(0), entries.at(1), aBegin, buldge);
  // buldge chasing
  for (int k = aBegin + 1; k < aEnd - 1; ++k) {
    entries = GetGivensEntries<>(matrix(k, k - 1), buldge);
    ApplyGivensTransformation<false, false, true, DataType>(matrix, entries.at(0), entries.at(1), k, buldge);
  }
  entries = GetGivensEntries<>(matrix(aEnd - 1, aEnd - 2), buldge);
  ApplyGivensTransformation<false, true, true, DataType>(matrix, entries.at(0), entries.at(1), aEnd - 1, buldge);

  return;
}


/* get the eigenvalues of a hessenberg matrix using the qr iteration
 * parameter:
 * - a_matrix: hessenberg matrix
 * - ak_is_hermitian: "true" if a_matrix is symmetric, "false" otherwise
 * - ak_tol: tolerance for considering a value 0
 * return: unordered vector of eigenvalues
 */
template <class DataType, bool is_hermitian, class matrix>
//std::enable_if_t<std::is_arithmetic<typename matrix::Scalar>::value && is_hermitian, std::vector<DataType>>
std::enable_if_t<std::is_arithmetic<typename ElementType<matrix>::Type>::value && is_hermitian, std::vector<DataType>>
QrIterationHessenberg(matrix &a_matrix,
                      const double ak_tol = 1e-12) {
  //EASY_FUNCTION(profiler::colors::Red);
  // generell definitions
  int begin = 0;
  int end = a_matrix.rows() - 1;

  // qr iteration
  while (0 < end) {
    //EASY_BLOCK("1 step including deflation of qr", profiler::colors::Green);
    int status = DeflateDiagonal<matrix>(a_matrix, begin, end, ak_tol);
    if (status > 1) {
      end = begin - 1;
      begin = 0;
    } else {
      //EASY_BLOCK("1 step of the qr iteration", profiler::colors::Yellow);
      //ImplicitQrStep<typename matrix::Scalar, is_hermitian, matrix>(a_matrix, begin, end);
      ImplicitQrStep<typename ElementType<matrix>::Type, is_hermitian, matrix>(a_matrix, begin, end);
      //EASY_END_BLOCK;
    }
  }
  return CalcEigenvaluesFromSchur<DataType, true>(a_matrix, true);
}

template <
    bool IsHermitian, typename Matrix,
    //class DataType = typename DoubleType<IsComplex<typename Matrix::Scalar>()>::type,
    class DataType = typename DoubleType<IsComplex<typename ElementType<Matrix>::Type>()>::type,
    class ComplexDataType = typename EvType<IsComplex<DataType>(), DataType>::type>
inline std::enable_if_t<IsHermitian, std::vector<ComplexDataType>>
QrMethod(const Matrix &ak_matrix, const double ak_tol = 1e-12) {
  //assert(ak_matrix.rows() == ak_matrix.cols());
  //static_assert(std::is_convertible<typename Matrix::Scalar, DataType>::value,
  static_assert(std::is_convertible<typename ElementType<Matrix>::Type, DataType>::value,
                "Matrix Elements must be convertible to DataType");
  Matrix A = ak_matrix;  // Do not change the input matrix
  return QrIterationHessenberg<ComplexDataType, IsHermitian>(A, ak_tol);
}

} // namespace nla_exam

#endif
