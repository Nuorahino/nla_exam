#ifndef QR_HH_
#define QR_HH_


#include <cassert>
#include <cmath>
#include <complex>
#include <iostream>
#include <random>
#include <type_traits>
#include <vector>

#include <eigen3/Eigen/Dense>

#include "helpfunctions.hh"
#include "symm_qr.hh"

namespace nla_exam {
/* Compute the Householder Vector
 * Parameter:
 * - ak_x: Vector to transform to multiple of e1
 * - a_tol: double, under which the tail is considered 0
 * Return: Householder Vector
 */
template <class Derived, class T = typename Derived::Scalar>
Eigen::Vector<T, -1>
GetHouseholderVector(const Eigen::MatrixBase<Derived> &ak_x) {
  Eigen::Vector<T, -1> w = ak_x;
  int64_t n = w.rows();
  T t = w(Eigen::lastN(n - 1)).squaredNorm();
  // TODO (Georg): Better Criteria needed
  if (std::abs(t) < std::numeric_limits<decltype(std::abs(t))>::min()) {
    w(0) = 1;
  } else {
    T s = std::sqrt(std::abs(w(0)) * std::abs(w(0)) + t);
    if constexpr (IsComplex<typename Derived::Scalar>()) {
      s *= w(0) / std::abs(w(0));  // Choise to avoid loss of significance
      // s *= std::polar(1.0, std::arg(w(0)));  // Does not work for float and double at
      // the same time
    } else {
      if (w(0) < 0) s *= -1;
    }
    w(0) = w(0) + s;
  }
  return w;
}


// TODO(Georg): Decide on if to include beta or not
/* Apply a Householder Reflection from the right
 * Parameter:
 * - ak_w: Householder Vector
 * - a_matrix: Matrix (Slice) to Transform
 * Return: void
 */
template <class Derived, class Derived2>
void
ApplyHouseholderRight(const Eigen::MatrixBase<Derived2> &ak_w,
                      const Eigen::MatrixBase<Derived> &a_matrix) {
  typedef typename Derived::Scalar T;
  typedef Eigen::MatrixBase<Derived> MatrixType;
  MatrixType &matrix = const_cast<MatrixType &>(a_matrix);  // Const cast needed for eigen

  T beta = 2 / ak_w.squaredNorm();
  for (int i = 0; i < a_matrix.rows(); ++i) {
    T tmp = beta * a_matrix(i, Eigen::all) * ak_w;
    matrix(i, Eigen::all) -= tmp * ak_w.adjoint();
  }
  return;
}


template <class Derived, class Derived2>
void
ApplyHouseholderRight(const Eigen::MatrixBase<Derived2> &ak_w,
                      const Eigen::MatrixBase<Derived> &a_matrix,
                      const typename Derived::Scalar beta) {
  typedef typename Derived::Scalar T;
  typedef Eigen::MatrixBase<Derived> MatrixType;
  MatrixType &matrix = const_cast<MatrixType &>(a_matrix);  // Const cast needed for eigen

  // T beta = 2 / ak_w.squaredNorm();
  for (int i = 0; i < a_matrix.rows(); ++i) {
    T tmp = beta * a_matrix(i, Eigen::all) * ak_w;
    matrix(i, Eigen::all) -= tmp * ak_w.adjoint();
  }
  return;
}


/* Apply a Householder Reflection from the left
 * Parameter:
 * - ak_w: Householder Vector
 * - a_matrix: Matrix (Slice) to Transform
 * Return: void
 */
template <class Derived, class Derived2>
void
ApplyHouseholderLeft(const Eigen::MatrixBase<Derived2> &ak_w,
                     const Eigen::MatrixBase<Derived> &a_matrix) {
  typedef typename Derived::Scalar T;
  typedef Eigen::MatrixBase<Derived> MatrixType;
  MatrixType &matrix = const_cast<MatrixType &>(a_matrix);  // Const cast needed for eigen

  T beta = 2 / ak_w.squaredNorm();
  for (int i = 0; i < a_matrix.cols(); ++i) {
    T tmp = beta * ak_w.dot(a_matrix(Eigen::all, i));  // w.dot(A) = w.adjoint() * A
    matrix(Eigen::all, i) -= tmp * ak_w;
  }
  return;
}


template <class Derived, class Derived2>
void
ApplyHouseholderLeft(const Eigen::MatrixBase<Derived2> &ak_w,
                     const Eigen::MatrixBase<Derived> &a_matrix,
                     const typename Derived::Scalar beta) {
  typedef typename Derived::Scalar T;
  typedef Eigen::MatrixBase<Derived> MatrixType;
  MatrixType &matrix = const_cast<MatrixType &>(a_matrix);  // Const cast needed for eigen

  // T beta = 2 / ak_w.squaredNorm();
  for (int i = 0; i < a_matrix.cols(); ++i) {
    T tmp = beta * ak_w.dot(a_matrix(Eigen::all, i));  // w.dot(A) = w.adjoint() * A
    matrix(Eigen::all, i) -= tmp * ak_w;
  }
  return;
}


/* Transforms a Matrix to Hessenberg form
 * Parameter:
 * - a_matrix: Matrix to transform
 * - ak_is_hermitian: bool
 * Return: void
 */
template <class Derived>
void
HessenbergTransformation(const Eigen::MatrixBase<Derived> &a_matrix,
                         const bool ak_is_hermitian = false) {
  typedef Eigen::MatrixBase<Derived> MatrixType;
  MatrixType &matrix = const_cast<MatrixType &>(a_matrix);
  int64_t n = a_matrix.rows();

  for (int i = 0; i < matrix.rows() - 2; ++i) {
    Eigen::Vector<typename Derived::Scalar, -1> w =
        GetHouseholderVector(matrix(Eigen::lastN(n - i - 1), i));
    ApplyHouseholderRight(w, matrix(Eigen::all, Eigen::lastN(n - i - 1)));
    ApplyHouseholderLeft(w, matrix(Eigen::lastN(n - i - 1), Eigen::seq(i, n - 1)));
    matrix(Eigen::seqN(i + 2, n - i - 2), i) = MatrixType::Zero(n - i - 2, 1);
    if (ak_is_hermitian) {
      matrix(i, Eigen::seqN(i + 2, n - i - 2)) = MatrixType::Zero(1, n - i - 2);
    }
  }
  if constexpr (IsComplex<typename Derived::Scalar>()) {
    // Transform complex Hermitian Matrix to a Real Tridiagonal Matrix
    if (ak_is_hermitian) {
      for (int i = 1; i < a_matrix.rows(); ++i) {
        // TODO (Georg): find condition for good sign
        matrix(i - 1, i) = std::abs(a_matrix(i - 1, i));
        matrix(i, i - 1) = std::abs(a_matrix(i, i - 1));
      }
    }
  }
  return;
}


/* Get Wilkinson shift parameter for a given 2x2 Matrix
 * Parameter:
 * - ak_matrix: 2x2 Matrix of which to calculate the shift parameter
 * Return: value of the shift parameter
 */

//// TODO(Georg): Check this is only meant to be called for real Eigenvalues?
//template <class DataType, bool is_symmetric, class Derived>
//inline std::enable_if_t<!is_symmetric && std::is_arithmetic<DataType>::value, DataType>
//WilkinsonShift(const Eigen::MatrixBase<Derived> &ak_matrix) {
//  DataType d = (ak_matrix(0, 0) - ak_matrix(1, 1)) / static_cast<DataType>(2.0);
//  // TODO (Georg): Find a way to compute this which avoids over and underflow
//  if (d >= 0) {
//    return ak_matrix(1, 1) + d - std::sqrt(d * d + ak_matrix(1, 0) * ak_matrix(0, 1));
//  } else {
//    return ak_matrix(1, 1) + d + std::sqrt(d * d + ak_matrix(1, 0) * ak_matrix(0, 1));
//  }
//}


template <class DataType, bool is_symmetric, class Matrix>
std::enable_if_t<IsComplex<DataType>(), DataType> inline
WilkinsonShift(const Matrix &ak_matrix, const int aEnd) {
  DataType trace = ak_matrix(aEnd, aEnd) + ak_matrix(aEnd - 1, aEnd - 1);
  DataType det = ak_matrix(aEnd, aEnd) * ak_matrix(aEnd - 1, aEnd - 1)
                  - ak_matrix(aEnd - 1, aEnd) * ak_matrix(aEnd, aEnd - 1);
  // TODO (Georg): Find a way to compute this which avoids over and underflow
  DataType tmp =
      std::sqrt(trace * trace - static_cast<DataType>(4.0) * det);
  DataType ev1 = (trace + tmp) / static_cast<DataType>(2.0);
  DataType ev2 = (trace - tmp) / static_cast<DataType>(2.0);

  DataType entry;
  entry = ak_matrix(1, 1);
  if (std::abs(ev1 - entry) < std::abs(ev2 - entry)) {  // return the nearest eigenvalue
    return ev1;
  } else {
    return ev2;
  }
}


///* Apply a Givens Rotation from the left
// * Parameter:
// * - a_matrix: Matrix (Slice) to transform
// * - ak_c: Givens parameter
// * - ak_s: Givens parameter
// * Return: void
// */
//template <class DataType, class Derived>
//std::enable_if_t<std::is_arithmetic<DataType>::value, void>
//ApplyGivensLeft(const Eigen::MatrixBase<Derived> &a_matrix, const DataType ak_c,
//                const DataType ak_s) {
//  Eigen::MatrixBase<Derived> &matrix = const_cast<Eigen::MatrixBase<Derived> &>(a_matrix);
//  for (int64_t i = 0; i < a_matrix.cols(); ++i) {
//    typename Derived::Scalar tmp = matrix(0, i);
//    matrix(0, i) = ak_c * tmp + ak_s * matrix(1, i);
//    matrix(1, i) = -ak_s * tmp + ak_c * matrix(1, i);
//  }
//  return;
//}
//
//
///* Apply a Givens Rotation from the right
// * Parameter:
// * - a_matrix: Matrix (slice) to transform
// * - ak_c: Givens parameter
// * - ak_s: Givens parameter
// * Return: void
// */
//template <class DataType, class Derived>
//std::enable_if_t<std::is_arithmetic<DataType>::value, void>
//ApplyGivensRight(const Eigen::MatrixBase<Derived> &a_matrix, const DataType ak_c,
//                 const DataType ak_s) {
//  Eigen::MatrixBase<Derived> &matrix = const_cast<Eigen::MatrixBase<Derived> &>(a_matrix);
//  for (int64_t i = 0; i < a_matrix.rows(); ++i) {
//    typename Derived::Scalar tmp = matrix(i, 0);
//    matrix(i, 0) = ak_c * tmp + ak_s * matrix(i, 1);
//    matrix(i, 1) = -ak_s * tmp + ak_c * matrix(i, 1);
//  }
//  return;
//}


/* Apply a Givens Rotation from the left
 * Parameter:
 * - a_matrix: Matrix (Slice) to transform
 * - ak_c: Givens parameter
 * - ak_s: Givens parameter
 * Return: void
 */
template <bool first, class DataType, class Matrix>
std::enable_if_t<!first, void>
ApplyGivensLeft(Matrix &matrix, const DataType ak_c,
                 const DataType ak_s, const DataType ak_sconj,
                 const int k, const int aEnd) {
  for (int64_t i = k - 1; i <= aEnd; ++i) {
    DataType tmp = matrix(k, i);
    matrix(k, i) = ak_c * tmp + ak_s * matrix(k + 1, i);
    matrix(k + 1, i) = -ak_sconj * tmp + ak_c * matrix(k + 1, i);
  }
  return;
}


template <bool first, class DataType, class Matrix>
std::enable_if_t<first, void>
ApplyGivensLeft(Matrix &matrix, const DataType ak_c,
                 const DataType ak_s, const DataType ak_sconj,
                 const int k, const int aEnd) {
  for (int64_t i = k; i <= aEnd; ++i) {
    DataType tmp = matrix(k, i);
    matrix(k, i) = ak_c * tmp + ak_s * matrix(k + 1, i);
    matrix(k + 1, i) = -ak_sconj * tmp + ak_c * matrix(k + 1, i);
  }
  return;
}


/* Apply a Givens Rotation from the right
 * Parameter:
 * - a_matrix: Matrix (slice) to transform
 * - ak_c: Givens parameter
 * - ak_s: Givens parameter
 * Return: void
 */
template <bool last, class DataType, class Matrix>
std::enable_if_t<last, void>
ApplyGivensRight(Matrix &matrix, const DataType ak_c,
                 const DataType ak_s, const DataType ak_sconj,
                 const int k, const int aBegin) {
  for (int64_t i = aBegin; i <= k + 1; ++i) {
    DataType tmp = matrix(i, k);
    matrix(i, k) = ak_c * tmp + ak_sconj * matrix(i, k + 1);
    matrix(i, k + 1) = -ak_s * tmp + ak_c * matrix(i, k + 1);
  }
  return;
}
//ApplyGivensRight(Matrix &matrix, const DataType ak_c,
//                 const DataType ak_s, const DataType ak_sconj,
//                 const int k, const int aBegin) {
//  for (int64_t i = aBegin; i <= k + 1; ++i) {
//    DataType tmp = matrix(i, k);
//    matrix(i, k) = ak_c * tmp + ak_sconj * matrix(i, k + 1);
//    matrix(i, k + 1) = -ak_s * tmp + ak_c * matrix(i, k + 1);
//  }
//  return;
//}

template <bool last, class DataType, class Matrix>
std::enable_if_t<!last, void>
ApplyGivensRight(Matrix &matrix, const DataType ak_c,
                 const DataType ak_s, const DataType ak_sconj,
                 const int k, const int aBegin) {
  for (int64_t i = aBegin; i <= k + 2; ++i) {
    DataType tmp = matrix(i, k);
    matrix(i, k) = ak_c * tmp + ak_sconj * matrix(i, k + 1);
    matrix(i, k + 1) = -ak_s * tmp + ak_c * matrix(i, k + 1);
  }
  return;
}


template <bool first, bool last, bool is_symmetric, class DataType, class Matrix>
inline std::enable_if_t<!std::is_arithmetic<DataType>::value || !is_symmetric, void>
ApplyGivensTransformation(Matrix &matrix, const int k, const int aBegin, const int aEnd, const DataType ak_c,
                const DataType ak_s, const DataType ak_sconj) {
  ApplyGivensLeft<first, DataType>(matrix, ak_c, ak_s, ak_sconj, k, aEnd);
  ApplyGivensRight<last, DataType>(matrix, ak_c, ak_s, ak_sconj, k, aBegin);
  return;
}

/* Calculate the entries of a Givens Matrix
 * Parameter:
 * - ak_a: first entry
 * - ak_b: entry to eliminate
 * Return: Vector containing {c, s, std::conj(s)}
 */
template <class DataType>
void
GetGivensEntries(const DataType &ak_a, const DataType &ak_b, std::array<DataType, 3> &res) {
  typedef typename DataType::value_type real;
  real absa = std::abs(ak_a);
  real absb = std::abs(ak_b);
  if (absa <= std::numeric_limits<typename DataType::value_type>::epsilon()) {
    res.at(0) = 0;
    res.at(1) = 1;
    res.at(2) = 1;
  } else {
    real r = std::hypot(absa, absb);
    res.at(0) = absa / r;
    res.at(1) = std::polar(absb / r, -std::arg(ak_b) + std::arg(ak_a));
    res.at(2) = std::conj(res.at(1));
  }
  //std::cout << "Own code" << std::endl;
  //std::cout << res.at(0) << ", " << res.at(1) << std::endl;
  //std::cout << "LAPACK" << std::endl;
//  std::array<DataType, 3> test;
//  compute_givens_parameter(ak_a, ak_b, test);
//  test.at(2) = std::conj(test.at(1));
  //std::cout << test.at(0) << ", " << test.at(1) << test.at(2) << std::endl;
  //std::cout << test << std::endl;
  return;
  //return res;
}


template <class DataType>
typename std::enable_if_t<std::is_arithmetic<DataType>::value, DataType>
ExceptionalSingleShift() {
  std::random_device dev;
  std::mt19937 rng(dev());
  std::uniform_real_distribution<DataType> dist(-100, 100);
  return dist(rng);
}

template <class DataType>
typename std::enable_if_t<!std::is_arithmetic<DataType>::value, DataType>
ExceptionalSingleShift() {
  std::random_device dev;
  std::mt19937 rng(dev());
  std::uniform_real_distribution<typename DataType::value_type> dist(-100, 100);
  return {dist(rng), dist(rng)};
}


/* Executes one step of the implicit qr algorithm for a tridiagonal Matrix
 * Parameter:
 * - a_matrix: Tridiagonal Matrix
 * Return: void
 */
template <class DataType, bool is_symmetric, typename Matrix>
std::enable_if_t<!std::is_arithmetic<DataType>::value || !is_symmetric, void>
ImplicitQrStep(Matrix &matrix, const int Begin, const int End, const bool ak_exceptional_shift) {
  DataType shift;
  if (ak_exceptional_shift) {
    shift = ExceptionalSingleShift<DataType>();
  } else {
    shift = WilkinsonShift<DataType, is_symmetric>(matrix, End);
  }
  std::array<DataType, 3> entries;
  GetGivensEntries<>(matrix(Begin, Begin) - shift, matrix(Begin + 1, Begin), entries);  // Parameter for the initial step
    // inital step
  if(End - Begin < 2) {
    ApplyGivensTransformation<true, true, false>(matrix, Begin, Begin, End, entries.at(0), entries.at(1), entries.at(2));
    return;
  }
  ApplyGivensTransformation<true, false, false>(matrix, Begin, Begin, End, entries.at(0), entries.at(1), entries.at(2));
  // buldge chasing
  for (int k = Begin + 1; k <= End - 2; ++k) {
    GetGivensEntries<>(matrix(k, k - 1), matrix(k + 1, k - 1), entries);
//    if (entries.size() == 2) {
//      entries.push_back(entries.at(1));  // needed for reel non symm matrix
//    }
    ApplyGivensTransformation<false, false, false>(matrix, k, Begin, End, entries.at(0), entries.at(1), entries.at(2));
    matrix(k + 1, k - 1) = 0.0;
  }

  GetGivensEntries<>(matrix(End - 1, End - 2), matrix(End, End - 2), entries);
  //if (entries.size() == 2) entries.push_back(entries.at(1));  // for reel non sym matrix
  ApplyGivensTransformation<false, true, false>(matrix, End - 1, Begin, End, entries.at(0), entries.at(1), entries.at(2));

  matrix(End, End - 2) = 0.0;
  return;
}


///* Get double shift parameters for a given 2x2 Matrix
// * Parameter:
// * - ak_matrix: 2x2 Matrix of which to calculate the shift parameter
// * Return: vector containing both double shift parameters
// */
//template <class Derived>
//typename std::enable_if_t<std::is_arithmetic<typename Derived::Scalar>::value,
//                          std::vector<typename Derived::Scalar>>
//DoubleShiftParameter(const Eigen::MatrixBase<Derived> &ak_matrix) {
//  typedef typename Derived::Scalar DataType;
//  std::vector<DataType> res(2);
//  res.at(0) = -ak_matrix.trace();
//  res.at(1) = ak_matrix.determinant();
//#ifdef SINGLE
//  if (res.at(0) * res.at(0) > 4.0 * res.at(1)) {
//#ifdef IMPLICIT
//    // For real eigenvalues do a single step
//    return std::vector<typename Derived::Scalar>{};
//#endif
//    DataType tmp = std::sqrt(res.at(0) * res.at(0) - 4.0 * res.at(1));
//    DataType ev1 = (-res.at(0) + tmp) / 2.0;
//    DataType ev2 = (-res.at(0) - tmp) / 2.0;
//    if (std::abs(ev1 - ak_matrix(1, 1)) < std::abs(ev2 - ak_matrix(1, 1))) {
//      res.at(0) = -2.0 * ev1;
//      res.at(1) = ev1 * ev1;
//    } else {
//      res.at(0) = -2.0 * ev2;
//      res.at(1) = ev2 * ev2;
//    }
//  }
//#endif
//  return res;
//}
//
//
///* Executes one step of the double shift algorithm
// * Parameter:
// * - a_matrix: Tridiagonal Matrix
// * Return: void
// */
//template <class Derived>
//void
//DoubleShiftQrStep(const Eigen::MatrixBase<Derived> &a_matrix,
//                  const bool ak_exceptional_shift) {
//  typedef Eigen::MatrixBase<Derived> MatrixType;
//  typedef Eigen::Matrix<typename Derived::Scalar, -1, -1> Matrix;
//  MatrixType &matrix = const_cast<MatrixType &>(a_matrix);
//  int n = a_matrix.rows();
//  std::vector<typename Derived::Scalar> shift;
//  if (ak_exceptional_shift) {
//    ImplicitQrStep<typename Derived::Scalar, false>(a_matrix, true);
//    return;
//  } else {
//    shift = DoubleShiftParameter<>(a_matrix(Eigen::lastN(2), Eigen::lastN(2)));
//  }
//  if (shift.size() == 0) {
//    ImplicitQrStep<typename Derived::Scalar, false>(a_matrix, false);
//    return;
//  }
//  // Only first three entries of first col needed
//  Matrix m1 = a_matrix(Eigen::seqN(0, 3), Eigen::all) * a_matrix(Eigen::all, 0) +
//              shift.at(0) * a_matrix(Eigen::seqN(0, 3), 0) +
//              shift.at(1) * Matrix::Identity(3, 1);
//
//  int64_t end = std::min(4, n);  // min needed for matrices of size 3
//  // Compute the Reflection for the first step of the shifted matrix
//  Eigen::Vector<typename Derived::Scalar, -1> w = GetHouseholderVector(m1);
//  ApplyHouseholderRight(w, matrix(Eigen::seqN(0, end), Eigen::seqN(0, 3)));
//  ApplyHouseholderLeft(w, matrix(Eigen::seqN(0, 3), Eigen::all));
//  // Buldge Chasing
//  for (int i = 0; i < n - 3; ++i) {
//    w = GetHouseholderVector(matrix(Eigen::seqN(i + 1, 3), i));
//    end = std::min(i + 4, n - 1);
//    ApplyHouseholderRight(w, matrix(Eigen::seq(0, end), Eigen::seqN(i + 1, 3)));
//    ApplyHouseholderLeft(w, matrix(Eigen::seqN(i + 1, 3), Eigen::seq(i, n - 1)));
//    matrix(Eigen::seqN(i + 2, 2), i) = Matrix::Zero(2, 1);  // Set Round off errors to 0
//  }
//  // TODO (Georg): Maybe Givens?
//  // last Reflection in the Buldge Chasing
//  w = GetHouseholderVector(matrix(Eigen::lastN(2), n - 3));
//  ApplyHouseholderRight(w, matrix(Eigen::all, Eigen::lastN(2)));
//  ApplyHouseholderLeft(w, matrix(Eigen::lastN(2), Eigen::lastN(3)));
//}
//
//
///* Deflates a Matrix converging to a Schur Matrix
// * Parameter:
// * - a_matrix: Matrix to deflate
// * - a_begin: Index of fhe block that is solved currently
// * - a_end: Index of fhe End that is solved currently
// * - ak_tol: Tolerance for considering a value 0
// * Return: "true" if the block is fully solved, "false" otherwise
// */
//template <class Derived>
//int
//DeflateSchur(const Eigen::MatrixBase<Derived> &a_matrix, int &a_begin, int &a_end,
//             const double ak_tol = 1e-12) {
//  int state = 2;
//  for (int i = a_end; i > a_begin; --i) {
//    if (std::abs(a_matrix(i, i - 1)) <=
//        (ak_tol * (std::abs(a_matrix(i, i)) + std::abs(a_matrix(i - 1, i - 1))))) {
//      const_cast<Eigen::MatrixBase<Derived> &>(a_matrix)(i, i - 1) = 0;
//      if (state < 2) {
//        if (i + 1 < a_end) {
//          a_begin = i;
//          return 1;  // Subblock to solve found
//        } else {
//          state = 2;
//        }
//      }
//    } else if (state == 2 && (i - 1 > a_begin)) {
//      if (i == a_end) {
//        state = 0;
//      } else {
//        a_end = i;
//        state = 1;
//      }
//    }
//  }
//  return state;
//}


/* Calculates the eigenvalues of a Matrix in Schur Form
 * Parameter:
 * - ak_matrix: Matrix in Schur Form
 * - ak_matrix_is_diagonal: "true" if ak_matrix is diagonal, "false" otherwise
 * Return: Unordered Vector of eigenvalues
 */
template <class DataType, bool is_symmetric, class Matrix>
inline std::enable_if_t<!is_symmetric, std::vector<DataType>>
CalcEigenvaluesFromSchur(const Matrix &ak_matrix,
                         const bool ak_matrix_is_diagonal = false) {
  int n = rows(ak_matrix);
  std::vector<DataType> res(n);
  if (ak_matrix_is_diagonal || n == 1) {
    for (int i = 0; i < n; ++i) {
      res.at(i) = ak_matrix(i, i);
    }
  } else {  // reel Schur form
    for (int i = 0; i < n - 1; ++i) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-equal"
      if (std::abs(ak_matrix(i + 1, i)) == 0) {  // Eigenvalue in diagonal block
#pragma GCC diagnostic pop
        res.at(i) = ak_matrix(i, i);
      } else {  // Eigenvalue in a 2x2 block
        DataType trace = ak_matrix(i, i) + ak_matrix(i + 1, i + 1);
        DataType det = ak_matrix(i, i) * ak_matrix(i + 1, i + 1) - ak_matrix(i + 1, i) * ak_matrix(i, i + 1);
        DataType tmp = std::sqrt(trace * trace - 4.0 * det);
        res.at(i) = (trace + tmp) / 2.0;
        ++i;
        res.at(i) = (trace - tmp) / 2.0;
      }
    }
#pragma GCC diagnostic ignored "-Wfloat-equal"
#pragma GCC diagnostic push
    // Last EV is in a diagonal block
    if (std::abs(ak_matrix(n - 1, n - 2)) == 0.0) {
#pragma GCC diagnostic pop
      // Add the last eigenvalue
      res.at(n - 1) = ak_matrix(n - 1, n - 1);
    }
  }
  return res;
}


/* Get the eigenvalues of a hessenberg Matrix using the qr iteration
 * Parameter:
 * - a_matrix: Hessenberg Matrix
 * - ak_is_hermitian: "true" if a_matrix is symmetric, "false" otherwise
 * - ak_tol: Tolerance for considering a value 0
 * Return: Unordered Vector of eigenvalues
 */
template <class DataType, bool is_hermitian, class Matrix>
std::enable_if_t<!std::is_arithmetic<typename ElementType<Matrix>::type>::value || !is_hermitian, std::vector<DataType>>
QrIterationHessenberg(Matrix &a_matrix, const double ak_tol = 1e-12) {
  typedef typename ElementType<Matrix>::type Scalar;
  // generell definitions
  int begin = 0;
  int end = rows<Matrix>(a_matrix) - 1;
  int steps_since_deflation = 0;

  // specific definitions
  int end_of_while = 0;
  bool tridiagonal_result = true;
  void (*step)(Matrix&, const int, const int, const bool);
  int (*deflate)(Matrix&, int &, int &, const double);
  // second part is only there in case of merge with symm file
  if constexpr (std::is_arithmetic<Scalar>::value && !is_hermitian) {
//    end_of_while = 1;
//    step = &DoubleShiftQrStep<Matrix>;
//    deflate = &DeflateSchur<Matrix>;
//    tridiagonal_result = false;
  } else {
    step = &ImplicitQrStep<Scalar, is_hermitian, Matrix>;
    deflate = &DeflateDiagonal<Matrix>;
  }

  // qr iteration
  while (end_of_while < end) {
    int status = deflate(a_matrix, begin, end, ak_tol);
    if (status > 0) {
      steps_since_deflation = 0;
      if (status > 1) {
        end = begin - 1;
        begin = 0;
      }
    } else {
      ++steps_since_deflation;
      bool exceptional_shift = false;
      if (steps_since_deflation > 10) {
        if constexpr (!is_hermitian) {
          exceptional_shift = true;
        }
        steps_since_deflation = 1;
      }
      step(a_matrix, begin, end, exceptional_shift);
    }
  }

  return CalcEigenvaluesFromSchur<DataType, false>(a_matrix, tridiagonal_result);
}


/* Calculate the eigenvalues of a Matrix using the QR decomposition
 * Parameter:
 * - a_matrix: Square Matrix
 * - ak_tol: Tolerance for considering a value 0
 * Return: Unordered Vector of (complex) eigenvalues
 */
template <
    bool IsHermitian, typename Matrix,
    class DataType = typename DoubleType<IsComplex<typename ElementType<Matrix>::type>()>::type,
    class ComplexDataType = typename EvType<IsComplex<DataType>(), DataType>::type>
inline std::enable_if_t<!IsHermitian, std::vector<ComplexDataType>>
QrMethod(const Matrix &ak_matrix, const double ak_tol = 1e-12) {
  //assert(ak_matrix.rows() == ak_matrix.cols());
  static_assert(std::is_convertible<typename ElementType<Matrix>::type, DataType>::value,
                "Matrix Elements must be convertible to DataType");
  Matrix A = ak_matrix;  // Do not change the input matrix
  return QrIterationHessenberg<ComplexDataType, IsHermitian>(A, ak_tol);
}
}  // namespace nla_exam
#endif
