#ifndef QR_HH_
#define QR_HH_

/*
 * TODO: optimize for eigen (noalias)
 * TODO: use threads
 */

#include <cassert>
#include <cmath>
#include <complex>
#include <iostream>
#include <random>
#include <type_traits>
#include <vector>

#include <easy/profiler.h>

#include <eigen3/Eigen/Dense>

#include "helpfunctions.hh"

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
  EASY_FUNCTION(profiler::colors::Red);
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
  EASY_FUNCTION(profiler::colors::Red);
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
  EASY_FUNCTION(profiler::colors::Red);
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
  EASY_FUNCTION(profiler::colors::Red);
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
  EASY_FUNCTION(profiler::colors::Red);
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
  EASY_FUNCTION(profiler::colors::Red);
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
template <class DataType, bool is_symmetric, class Derived>
inline std::enable_if_t<is_symmetric && std::is_arithmetic<DataType>::value, DataType>
WilkinsonShift(const Eigen::MatrixBase<Derived> &ak_matrix) {
  EASY_FUNCTION(profiler::colors::Red);
  DataType d = (ak_matrix(0, 0) - ak_matrix(1, 1)) / static_cast<DataType>(2.0);
  if (d >= 0) {
    return ak_matrix(1, 1) + d - std::hypot(d, ak_matrix(1, 0));
  } else {
    return ak_matrix(1, 1) + d + std::hypot(d, ak_matrix(1, 0));
  }
}


// TODO(Georg): Check this is only meant to be called for real Eigenvalues?
template <class DataType, bool is_symmetric, class Derived>
inline std::enable_if_t<!is_symmetric && std::is_arithmetic<DataType>::value, DataType>
WilkinsonShift(const Eigen::MatrixBase<Derived> &ak_matrix) {
  EASY_FUNCTION(profiler::colors::Red);
  DataType d = (ak_matrix(0, 0) - ak_matrix(1, 1)) / static_cast<DataType>(2.0);
  // TODO (Georg): Find a way to compute this which avoids over and underflow
  if (d >= 0) {
    return ak_matrix(1, 1) + d - std::sqrt(d * d + ak_matrix(1, 0) * ak_matrix(0, 1));
  } else {
    return ak_matrix(1, 1) + d + std::sqrt(d * d + ak_matrix(1, 0) * ak_matrix(0, 1));
  }
}


template <class DataType, bool is_symmetric, class Derived>
std::enable_if_t<IsComplex<DataType>(), DataType>
WilkinsonShift(const Eigen::MatrixBase<Derived> &ak_matrix) {
  EASY_FUNCTION(profiler::colors::Red);
  DataType trace = ak_matrix.trace();
  // TODO (Georg): Find a way to compute this which avoids over and underflow
  DataType tmp =
      std::sqrt(trace * trace - static_cast<DataType>(4.0) * ak_matrix.determinant());
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

template <bool first, bool last, class DataType, class Derived>
std::enable_if_t<std::is_arithmetic<DataType>::value, void>
ApplyGivensTransformation(const Eigen::MatrixBase<Derived> &a_matrix, const DataType ak_c,
                const DataType ak_s, const int aBegin) {
  Eigen::MatrixBase<Derived> &matrix = const_cast<Eigen::MatrixBase<Derived> &>(a_matrix);
  int k = aBegin;
  DataType c = ak_c;
  DataType s = ak_s;
  DataType x1 = c * c * matrix(k, k) + s * s * matrix(k+1,k+1) + 2 * c * s * matrix(k, k+1);
  DataType a2 = c * -s * matrix(k,k) - s * s * matrix(k, k+1) + c * c * matrix(k, k+1) + s * c * matrix(k+1, k+1);
  DataType x2 = c * c * matrix(k+1, k+1) + s * s * matrix(k,k) - 2 * c * s * matrix(k+1, k);

  if constexpr (!first) {
    DataType a1 = c * matrix(k, k - 1) + s * a_matrix(k+1, k - 1);
    matrix(k-1, k) = a1;
    matrix(k, k-1) = a1;
  }
  if constexpr (!last) {
    DataType e = s * matrix(k+2, k+1);
    matrix(k, k+2) = e;
    matrix(k+2, k) = e;
    DataType a3 = c * matrix(k+1, k+2);
    matrix(k+1, k+2) = a3;
    matrix(k+2, k+1) = a3;
  }

  matrix(k+1, k) = a2;
  matrix(k, k+1) = a2;
  matrix(k, k) = x1;
  matrix(k+1, k+1) = x2;

  matrix(k - 1, k + 1) = 0;
  matrix(k + 1, k - 1) = 0;
  return;
}

/* Apply a Givens Rotation from the left
 * Parameter:
 * - a_matrix: Matrix (Slice) to transform
 * - ak_c: Givens parameter
 * - ak_s: Givens parameter
 * Return: void
 */
template <class DataType, class Derived>
std::enable_if_t<std::is_arithmetic<DataType>::value, void>
ApplyGivensLeft(const Eigen::MatrixBase<Derived> &a_matrix, const DataType ak_c,
                const DataType ak_s) {
  EASY_FUNCTION(profiler::colors::Red);
  Eigen::MatrixBase<Derived> &matrix = const_cast<Eigen::MatrixBase<Derived> &>(a_matrix);
  for (int64_t i = 0; i < a_matrix.cols(); ++i) {
    typename Derived::Scalar tmp = matrix(0, i);
    matrix(0, i) = ak_c * tmp + ak_s * matrix(1, i);
    matrix(1, i) = -ak_s * tmp + ak_c * matrix(1, i);
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
template <class DataType, class Derived>
std::enable_if_t<std::is_arithmetic<DataType>::value, void>
ApplyGivensRight(const Eigen::MatrixBase<Derived> &a_matrix, const DataType ak_c,
                 const DataType ak_s) {
  EASY_FUNCTION(profiler::colors::Red);
  Eigen::MatrixBase<Derived> &matrix = const_cast<Eigen::MatrixBase<Derived> &>(a_matrix);
  for (int64_t i = 0; i < a_matrix.rows(); ++i) {
    typename Derived::Scalar tmp = matrix(i, 0);
    matrix(i, 0) = ak_c * tmp + ak_s * matrix(i, 1);
    matrix(i, 1) = -ak_s * tmp + ak_c * matrix(i, 1);
  }
  return;
}


/* Apply a Givens Rotation from the left
 * Parameter:
 * - a_matrix: Matrix (Slice) to transform
 * - ak_c: Givens parameter
 * - ak_s: Givens parameter
 * Return: void
 */
template <class DataType, class Derived>
void
ApplyGivensLeft(const Eigen::MatrixBase<Derived> &a_matrix, const DataType ak_c,
                const DataType ak_s, const DataType ak_sconj) {
  EASY_FUNCTION(profiler::colors::Red);
  Eigen::MatrixBase<Derived> &matrix = const_cast<Eigen::MatrixBase<Derived> &>(a_matrix);
  for (int64_t i = 0; i < a_matrix.cols(); ++i) {
    typename Derived::Scalar tmp = matrix(0, i);
    matrix(0, i) = ak_c * tmp + ak_s * matrix(1, i);
    matrix(1, i) = -ak_sconj * tmp + ak_c * matrix(1, i);
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
template <class DataType, class Derived>
void
ApplyGivensRight(const Eigen::MatrixBase<Derived> &a_matrix, const DataType ak_c,
                 const DataType ak_s, const DataType ak_sconj) {
  EASY_FUNCTION(profiler::colors::Red);
  Eigen::MatrixBase<Derived> &matrix = const_cast<Eigen::MatrixBase<Derived> &>(a_matrix);
  for (int64_t i = 0; i < a_matrix.rows(); ++i) {
    typename Derived::Scalar tmp = matrix(i, 0);
    matrix(i, 0) = ak_c * tmp + ak_sconj * matrix(i, 1);
    matrix(i, 1) = -ak_s * tmp + ak_c * matrix(i, 1);
  }
  return;
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
  EASY_FUNCTION(profiler::colors::Red);
  std::vector<DataType> res(3);
  if (std::abs(ak_a) <= std::numeric_limits<DataType>::epsilon()) {
    res.at(0) = 0;
    res.at(1) = 1;
  } else {
    DataType r = std::hypot(ak_a, ak_b);
    res.at(0) = std::abs(ak_a) / r;
    res.at(1) =
        ak_b / r *
        DataType{std::copysign(
            DataType{1}, ak_a)};  // TODO: instead of copysign maybe use a test with >
                                  // 0 to do this, as this always converts to float
  }
  res.at(2) = res.at(1);
  return res;
}


/* Calculate the entries of a Givens Matrix
 * Parameter:
 * - ak_a: first entry
 * - ak_b: entry to eliminate
 * Return: Vector containing {c, s, std::conj(s)}
 */
template <class DataType>
inline std::enable_if_t<IsComplex<DataType>(), std::vector<DataType>>
GetGivensEntries(const DataType &ak_a, const DataType &ak_b) {
  EASY_FUNCTION(profiler::colors::Red);
  typedef typename DataType::value_type real;
  std::vector<DataType> res(3);
  real absa = std::abs(ak_a);
  real absb = std::abs(ak_b);
  if (absa <= std::numeric_limits<typename DataType::value_type>::epsilon()) {
    res.at(0) = 0;
    res.at(1) = 1;
    res.at(1) = 1;
  } else {
    real r = std::hypot(absa, absb);
    res.at(0) = absa / r;
    res.at(1) = std::polar(absb / r, -std::arg(ak_b) + std::arg(ak_a));
    res.at(2) = std::conj(res.at(1));
  }
  return res;
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
template <class DataType, bool is_symmetric, typename Derived>
void
ImplicitQrStep(const Eigen::MatrixBase<Derived> &a_matrix,
               const bool ak_exceptional_shift) {
  EASY_FUNCTION(profiler::colors::Red);
  Eigen::MatrixBase<Derived> &matrix = const_cast<Eigen::MatrixBase<Derived> &>(a_matrix);
  int n = a_matrix.rows();
  DataType shift;
  if (ak_exceptional_shift) {
    shift = ExceptionalSingleShift<DataType>();
  } else {
    shift = WilkinsonShift<DataType, is_symmetric>(
        a_matrix(Eigen::lastN(2), Eigen::lastN(2)));
  }
  auto entries = GetGivensEntries<>(a_matrix(0, 0) - shift,
                                    a_matrix(1, 0));  // Parameter for the initial step
  if constexpr (is_symmetric) {  // TODO (Georg): maybe better solution if matrix has
                                 // symmetric view
    // innitial step
    switch (n) {
      case 2:
        ApplyGivensLeft<DataType>(matrix, entries.at(0), entries.at(1));
        ApplyGivensRight<DataType>(matrix, entries.at(0), entries.at(1));
        return;
      default:
        ApplyGivensLeft<DataType>(matrix(Eigen::seqN(0, 2), Eigen::seqN(0, 3)),
                                  entries.at(0), entries.at(1));
        ApplyGivensRight<DataType>(matrix(Eigen::seqN(0, 3), Eigen::seqN(0, 2)),
                                   entries.at(0), entries.at(1));
    }
    // buldge chasing
    for (int k = 1; k < n - 2; ++k) {
      entries = GetGivensEntries<>(a_matrix(k, k - 1), a_matrix(k + 1, k - 1));
      ApplyGivensLeft<DataType>(matrix(Eigen::seqN(k, 2), Eigen::seq(k - 1, n - 1)),
                                entries.at(0), entries.at(1));
      ApplyGivensRight<DataType>(matrix(Eigen::seq(0, k + 2), Eigen::seqN(k, 2)),
                                 entries.at(0), entries.at(1));
      matrix(k - 1, k + 1) = 0;
      matrix(k + 1, k - 1) = 0;
    }
    entries = GetGivensEntries<>(a_matrix(n - 2, n - 3), a_matrix(n - 1, n - 3));
    ApplyGivensLeft<DataType>(matrix(Eigen::seqN(n - 2, 2), Eigen::lastN(3)),
                              entries.at(0), entries.at(1));
    ApplyGivensRight<DataType>(matrix(Eigen::all, Eigen::seqN(n - 2, 2)), entries.at(0),
                               entries.at(1));
    matrix(n - 3, n - 1) = 0;
    matrix(n - 1, n - 3) = 0;

  } else {
    // inital step
    switch (n) {
      case 2:
        ApplyGivensLeft<DataType>(matrix, entries.at(0), entries.at(1), entries.at(2));
        ApplyGivensRight<DataType>(matrix, entries.at(0), entries.at(1), entries.at(2));
        return;
      default:
        ApplyGivensLeft<DataType>(matrix(Eigen::seqN(0, 2), Eigen::all), entries.at(0),
                                  entries.at(1), entries.at(2));
        ApplyGivensRight<DataType>(matrix(Eigen::seqN(0, 3), Eigen::seqN(0, 2)),
                                   entries.at(0), entries.at(1), entries.at(2));
    }
    // buldge chasing
    for (int k = 1; k < n - 2; ++k) {
      entries = GetGivensEntries<>(a_matrix(k, k - 1), a_matrix(k + 1, k - 1));
      if (entries.size() == 2) {
        entries.push_back(entries.at(1));  // needed for reel non symm matrix
      }
      ApplyGivensLeft<DataType>(matrix(Eigen::seqN(k, 2), Eigen::seq(k - 1, n - 1)),
                                entries.at(0), entries.at(1), entries.at(2));
      ApplyGivensRight<DataType>(matrix(Eigen::seq(0, k + 2), Eigen::seqN(k, 2)),
                                 entries.at(0), entries.at(1), entries.at(2));
      matrix(k + 1, k - 1) = 0.0;
    }
    entries = GetGivensEntries<>(a_matrix(n - 2, n - 3), a_matrix(n - 1, n - 3));
    if (entries.size() == 2) entries.push_back(entries.at(1));  // for reel non sym matrix
    ApplyGivensLeft<DataType>(matrix(Eigen::seqN(n - 2, 2), Eigen::lastN(3)),
                              entries.at(0), entries.at(1), entries.at(2));
    ApplyGivensRight<DataType>(matrix(Eigen::all, Eigen::seqN(n - 2, 2)), entries.at(0),
                               entries.at(1), entries.at(2));
    matrix(n - 1, n - 3) = 0.0;
  }
  return;
}


/* Get double shift parameters for a given 2x2 Matrix
 * Parameter:
 * - ak_matrix: 2x2 Matrix of which to calculate the shift parameter
 * Return: vector containing both double shift parameters
 */
template <class Derived>
typename std::enable_if_t<std::is_arithmetic<typename Derived::Scalar>::value,
                          std::vector<typename Derived::Scalar>>
DoubleShiftParameter(const Eigen::MatrixBase<Derived> &ak_matrix) {
  EASY_FUNCTION(profiler::colors::Red);
  typedef typename Derived::Scalar DataType;
  std::vector<DataType> res(2);
  res.at(0) = -ak_matrix.trace();
  res.at(1) = ak_matrix.determinant();
#ifdef SINGLE
  if (res.at(0) * res.at(0) > 4.0 * res.at(1)) {
#ifdef IMPLICIT
    // For real eigenvalues do a single step
    return std::vector<typename Derived::Scalar>{};
#endif
    DataType tmp = std::sqrt(res.at(0) * res.at(0) - 4.0 * res.at(1));
    DataType ev1 = (-res.at(0) + tmp) / 2.0;
    DataType ev2 = (-res.at(0) - tmp) / 2.0;
    if (std::abs(ev1 - ak_matrix(1, 1)) < std::abs(ev2 - ak_matrix(1, 1))) {
      res.at(0) = -2.0 * ev1;
      res.at(1) = ev1 * ev1;
    } else {
      res.at(0) = -2.0 * ev2;
      res.at(1) = ev2 * ev2;
    }
  }
#endif
  return res;
}


/* Executes one step of the double shift algorithm
 * Parameter:
 * - a_matrix: Tridiagonal Matrix
 * Return: void
 */
template <class Derived>
void
DoubleShiftQrStep(const Eigen::MatrixBase<Derived> &a_matrix,
                  const bool ak_exceptional_shift) {
  EASY_FUNCTION(profiler::colors::Red);
  typedef Eigen::MatrixBase<Derived> MatrixType;
  typedef Eigen::Matrix<typename Derived::Scalar, -1, -1> Matrix;
  MatrixType &matrix = const_cast<MatrixType &>(a_matrix);
  int n = a_matrix.rows();
  std::vector<typename Derived::Scalar> shift;
  if (ak_exceptional_shift) {
    ImplicitQrStep<typename Derived::Scalar, false>(a_matrix, true);
    return;
  } else {
    shift = DoubleShiftParameter<>(a_matrix(Eigen::lastN(2), Eigen::lastN(2)));
  }
  if (shift.size() == 0) {
    ImplicitQrStep<typename Derived::Scalar, false>(a_matrix, false);
    return;
  }
  // Only first three entries of first col needed
  Matrix m1 = a_matrix(Eigen::seqN(0, 3), Eigen::all) * a_matrix(Eigen::all, 0) +
              shift.at(0) * a_matrix(Eigen::seqN(0, 3), 0) +
              shift.at(1) * Matrix::Identity(3, 1);

  int64_t end = std::min(4, n);  // min needed for matrices of size 3
  // Compute the Reflection for the first step of the shifted matrix
  Eigen::Vector<typename Derived::Scalar, -1> w = GetHouseholderVector(m1);
  ApplyHouseholderRight(w, matrix(Eigen::seqN(0, end), Eigen::seqN(0, 3)));
  ApplyHouseholderLeft(w, matrix(Eigen::seqN(0, 3), Eigen::all));
  // Buldge Chasing
  for (int i = 0; i < n - 3; ++i) {
    w = GetHouseholderVector(matrix(Eigen::seqN(i + 1, 3), i));
    end = std::min(i + 4, n - 1);
    ApplyHouseholderRight(w, matrix(Eigen::seq(0, end), Eigen::seqN(i + 1, 3)));
    ApplyHouseholderLeft(w, matrix(Eigen::seqN(i + 1, 3), Eigen::seq(i, n - 1)));
    matrix(Eigen::seqN(i + 2, 2), i) = Matrix::Zero(2, 1);  // Set Round off errors to 0
  }
  // TODO (Georg): Maybe Givens?
  // last Reflection in the Buldge Chasing
  w = GetHouseholderVector(matrix(Eigen::lastN(2), n - 3));
  ApplyHouseholderRight(w, matrix(Eigen::all, Eigen::lastN(2)));
  ApplyHouseholderLeft(w, matrix(Eigen::lastN(2), Eigen::lastN(3)));
}


/* Deflates a Matrix converging to a diagonal matrix
 * Parameter:
 * - a_matrix: Matrix to deflate
 * - a_begin: Index of fhe block that is solved currently
 * - a_end: Index of fhe End that is solved currently
 * - ak_tol: Tolerance for considering a value 0
 * Return: "true" if the block is fully solved, "false" otherwise
 */
template <class Derived>
int
DeflateDiagonal(const Eigen::MatrixBase<Derived> &a_matrix, int &a_begin, int &a_end,
                const double ak_tol = 1e-12) {
  int state = 2;
  for (int i = a_end; i > a_begin; --i) {
    if (std::abs(a_matrix(i, i - 1)) <=
        (ak_tol * (std::abs(a_matrix(i, i)) + std::abs(a_matrix(i - 1, i - 1))))) {
      const_cast<Eigen::MatrixBase<Derived> &>(a_matrix)(i, i - 1) = 0;
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


/* Deflates a Matrix converging to a Schur Matrix
 * Parameter:
 * - a_matrix: Matrix to deflate
 * - a_begin: Index of fhe block that is solved currently
 * - a_end: Index of fhe End that is solved currently
 * - ak_tol: Tolerance for considering a value 0
 * Return: "true" if the block is fully solved, "false" otherwise
 */
template <class Derived>
int
DeflateSchur(const Eigen::MatrixBase<Derived> &a_matrix, int &a_begin, int &a_end,
             const double ak_tol = 1e-12) {
  int state = 2;
  for (int i = a_end; i > a_begin; --i) {
    if (std::abs(a_matrix(i, i - 1)) <=
        (ak_tol * (std::abs(a_matrix(i, i)) + std::abs(a_matrix(i - 1, i - 1))))) {
      const_cast<Eigen::MatrixBase<Derived> &>(a_matrix)(i, i - 1) = 0;
      if (state < 2) {
        if (i + 1 < a_end) {
          a_begin = i;
          return 1;  // Subblock to solve found
        } else {
          state = 2;
        }
      }
    } else if (state == 2 && (i - 1 > a_begin)) {
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


/* Calculates the eigenvalues of a Matrix in Schur Form
 * Parameter:
 * - ak_matrix: Matrix in Schur Form
 * - ak_matrix_is_diagonal: "true" if ak_matrix is diagonal, "false" otherwise
 * Return: Unordered Vector of eigenvalues
 */
template <class DataType, class Derived>
std::vector<DataType>
CalcEigenvaluesFromSchur(const Eigen::MatrixBase<Derived> &ak_matrix,
                         const bool ak_matrix_is_diagonal = false) {
  std::vector<DataType> res(ak_matrix.rows());
  if (ak_matrix_is_diagonal || ak_matrix.rows() == 1) {
    for (int i = 0; i < ak_matrix.rows(); ++i) {
      res.at(i) = ak_matrix(i, i);
    }
  } else {  // reel Schur form
    for (int i = 0; i < ak_matrix.rows() - 1; ++i) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-equal"
      if (std::abs(ak_matrix(i + 1, i)) == 0) {  // Eigenvalue in diagonal block
#pragma GCC diagnostic pop
        res.at(i) = ak_matrix(i, i);
      } else {  // Eigenvalue in a 2x2 block
        Eigen::MatrixXcd test = ak_matrix(Eigen::seq(i, i + 1), Eigen::seq(i, i + 1));
        DataType trace = test.trace();
        DataType tmp = std::sqrt(trace * trace - 4.0 * test.determinant());
        res.at(i) = (trace + tmp) / 2.0;
        ++i;
        res.at(i) = (trace - tmp) / 2.0;
      }
    }
#pragma GCC diagnostic ignored "-Wfloat-equal"
#pragma GCC diagnostic push
    // Last EV is in a diagonal block
    if (std::abs(ak_matrix(ak_matrix.rows() - 1, ak_matrix.rows() - 2)) == 0.0) {
#pragma GCC diagnostic pop
      // Add the last eigenvalue
      res.at(ak_matrix.rows() - 1) =
          ak_matrix(ak_matrix.rows() - 1, ak_matrix.rows() - 1);
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
template <class DataType, bool ak_is_hermitian, class Derived>
std::vector<DataType>
QrIterationHessenberg(const Eigen::MatrixBase<Derived> &a_matrix,
                      const double ak_tol = 1e-12) {
  EASY_PROFILER_ENABLE;
  EASY_BLOCK("QR iteration initalization", profiler::colors::Blue);
  // generell definitions
  typedef Eigen::MatrixBase<Derived> MatrixType;
  typedef Eigen::Block<Derived, -1, -1, false> StepMatrix;
  int begin = 0;
  int end = a_matrix.rows() - 1;
  int steps_since_deflation = 0;

  // specific definitions
  int end_of_while = 0;
  bool tridiagonal_result = true;
  void (*step)(const Eigen::MatrixBase<StepMatrix> &, bool);
  int (*deflate)(const Eigen::MatrixBase<Derived> &, int &, int &, const double);
  if constexpr (std::is_arithmetic<typename Derived::Scalar>::value && !ak_is_hermitian) {
    end_of_while = 1;
    step = &DoubleShiftQrStep<StepMatrix>;
    deflate = &DeflateSchur<Derived>;
    tridiagonal_result = false;
  } else {
    step = &ImplicitQrStep<typename MatrixType::Scalar, ak_is_hermitian, StepMatrix>;
    deflate = &DeflateDiagonal<Derived>;
  }
  EASY_END_BLOCK;

  // qr iteration
  while (end_of_while < end) {
    EASY_BLOCK("1 Step including deflation of QR", profiler::colors::Green);
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
        if constexpr (!ak_is_hermitian) {
          exceptional_shift = true;
        }
        steps_since_deflation = 1;
      }
      EASY_BLOCK("1 Step of the QR Iteration", profiler::colors::Yellow);
      step(const_cast<MatrixType &>(a_matrix)(Eigen::seq(begin, end),
                                              Eigen::seq(begin, end)),
           exceptional_shift);
      EASY_END_BLOCK;
      EASY_END_BLOCK;
    }
  }
  return CalcEigenvaluesFromSchur<DataType>(a_matrix, tridiagonal_result);
}


/* Calculate the eigenvalues of a Matrix using the QR decomposition
 * Parameter:
 * - a_matrix: Square Matrix
 * - ak_tol: Tolerance for considering a value 0
 * Return: Unordered Vector of (complex) eigenvalues
 */
template <
    bool IsHermitian, typename Derived,
    class DataType = typename DoubleType<IsComplex<typename Derived::Scalar>()>::type,
    class ComplexDataType = typename EvType<IsComplex<DataType>(), DataType>::type>
std::vector<ComplexDataType>
QrMethod(const Eigen::MatrixBase<Derived> &ak_matrix, const double ak_tol = 1e-12) {
  assert(ak_matrix.rows() == ak_matrix.cols());
  static_assert(std::is_convertible<typename Derived::Scalar, DataType>::value,
                "Matrix Elements must be convertible to DataType");
  typedef Eigen::Matrix<DataType, -1, -1> Matrix;
  Matrix A = ak_matrix;  // Do not change the input matrix
  return QrIterationHessenberg<ComplexDataType, IsHermitian>(A, ak_tol);
}
}  // namespace nla_exam
#endif
