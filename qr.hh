#ifndef QR_HH_
#define QR_HH_

/*
 * TODO: optimize for eigen (noalias)
 * TODO: include the deflation in the step ?
 * TODO: use threads
 */

#include <cmath>
#include <complex>
#include <type_traits>
#include <vector>
#include <iostream>
#include <cassert>
#include <random> // For Bernulli Distribution

#include <eigen3/Eigen/Dense>

#include "helpfunctions/helpfunctions.hh"

namespace nla_exam {
//  static std::random_device rd;
//  static std::mt19937 gen(rd());
//  static std::binomial_distribution<> d;

/* Apply a Householder Reflection
 * Parameter:
 * - ak_x: Vector to transform to multiple of e1
 * - a_matrix: Matrix to Transform
 * - a_start: Index of the first row of the Householder Vector
 * Return: void
 */

template<class Derived, class T = typename Derived::Scalar>
Eigen::Vector<T, -1> GetHouseholderVector(const Eigen::MatrixBase<Derived> &ak_x,
                          const double ak_tol = 1e-12) {
  //typedef typename Derived::Scalar T;

  Eigen::Vector<T, -1> w = ak_x;
  long n = w.rows();
  T t = w(Eigen::lastN(n-1)).squaredNorm();
  if (std::abs(t) < ak_tol) {
    w(0) = 1;
    std::cout << "tail is zero" << std::endl;
  } else {
    T s = std::sqrt(w(0) * w(0) + t);
    T angle;
    if constexpr (IsComplex<typename Derived::Scalar>()) {
      angle = std::polar(1.0, arg(w(0)));                                       // Choise to avoid loss of significance
    } else {
      if (w(0) < 0) angle = -1;
      else angle = 1;
    }
    w(0) = w(0) + s * angle;
  }
  return w;
}


template<class Derived, class Derived2>
void ApplyHouseholderRight(const Eigen::MatrixBase<Derived2> &ak_w,
                           const Eigen::MatrixBase<Derived> &a_matrix) {
  typedef typename Derived::Scalar T;
  typedef Eigen::MatrixBase<Derived> MatrixType;
  MatrixType& matrix = const_cast<MatrixType&>(a_matrix);

  T beta = 2 / ak_w.squaredNorm();
  for(int i = 0; i < a_matrix.rows(); ++i) {
    T tmp = beta * a_matrix(i, Eigen::all) * ak_w;
    matrix(i, Eigen::all) -= tmp * ak_w.adjoint();
  }
  return;
}

template<class Derived, class Derived2>
void ApplyHouseholderLeft(const Eigen::MatrixBase<Derived2> &ak_w,
                          const Eigen::MatrixBase<Derived> &a_matrix) {
  typedef typename Derived::Scalar T;
  typedef Eigen::MatrixBase<Derived> MatrixType;
  MatrixType& matrix = const_cast<MatrixType&>(a_matrix);

  T beta = 2 / ak_w.squaredNorm();
  for(int i = 0; i < a_matrix.cols(); ++i) {
    T tmp = beta * ak_w.dot(a_matrix(Eigen::all, i));           // w.dot(A) = w.adjoint() * A
    matrix(Eigen::all,i) -= tmp * ak_w;
  }
  return;
}
template<class Derived, class Derived2>
void ApplyReverseHouseholder(const Eigen::MatrixBase<Derived2> &ak_x,
                             const Eigen::MatrixBase<Derived> &a_matrix,
                             const long a_start,
                             const long a_end,
                             const double ak_tol = 1e-12) {
  typedef typename Derived::Scalar T;
  typedef Eigen::MatrixBase<Derived> MatrixType;
  MatrixType& matrix = const_cast<MatrixType&>(a_matrix);
  Eigen::Vector<T, -1> w = ak_x;
  const long n = w.rows();
  T alpha = w.norm();
  if constexpr (IsComplex<typename Derived::Scalar>()) {
    alpha *= std::polar(1.0, arg(w(n - 1)));                                       // Choise to avoid loss of significance
  } else {
    if (w(n - 1) < 0) alpha *= -1;
  }
  w(n - 1) = ak_x(n - 1) + alpha;
  if (w.squaredNorm() < ak_tol) return;
  T beta = 2 / w.squaredNorm();
  for(int i = std::max(a_start - 2, long{0}) ; i < a_matrix.cols(); ++i) {
    alpha = beta * w.dot(a_matrix(Eigen::seqN(a_start, n),i));
    matrix(Eigen::seqN(a_start, n),i) -= alpha * w;
  }
  for(int i = 0; i < a_end; ++i) {
    alpha = beta * a_matrix(i, Eigen::seqN(a_start, n)) * w;
    matrix(i, Eigen::seqN(a_start, n)) -= alpha * w.adjoint().eval();
  }
}


/* Transforms a Matrix to Hessenberg form
 * Parameter:
 * - a_matrix: Matrix to transform
 * Return: The unitary matrix used in the similarity trasnformation
 */
template <class Derived>
void HessenbergTransformation(const Eigen::MatrixBase<Derived> &a_matrix,
                              const double ak_tol = 1e-12,
                              const bool a_is_hermitian = false) {
  typedef Eigen::MatrixBase<Derived> MatrixType;
  MatrixType& matrix = const_cast<MatrixType&>(a_matrix);
  long n = a_matrix.rows();

  for (int i = 0; i < matrix.rows() - 2; ++i) {
    Eigen::Vector<typename Derived::Scalar, -1> w = GetHouseholderVector(matrix(
                                  Eigen::lastN(n - i - 1), i), ak_tol);
    ApplyHouseholderRight(w, matrix(Eigen::all, Eigen::lastN(n - i - 1)));
    ApplyHouseholderLeft(w, matrix(Eigen::lastN(n - i - 1), Eigen::seq(i, n-1)));
    matrix(Eigen::seqN(i + 2, n - i - 2), i) = MatrixType::Zero(n - i - 2, 1);
  }
  if constexpr (IsComplex<typename Derived::Scalar>()) {
    if (a_is_hermitian) {                                                         // Transform complex Hermitian Matrix to Real
      for(int i = 1; i < a_matrix.rows(); ++i) {
        // TODO find condition for good sign
        matrix(i-1, i) = std::abs(a_matrix(i-1, i));
        matrix(i, i-1) = std::abs(a_matrix(i, i-1));
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
template <class DataType, class Derived> inline
std::enable_if_t<std::is_arithmetic<DataType>::value, DataType>
WilkinsonShift(const Eigen::MatrixBase<Derived> &ak_matrix, const bool a_first = true) {
  DataType d = (ak_matrix(0, 0) - ak_matrix(1, 1)) / 2.0;
  if ((d >=  0 && a_first) || (!a_first && d <= 0)) {
    return ak_matrix(1, 1) + d - std::hypot(d, ak_matrix(1,0));
  } else {
    return ak_matrix(1, 1) + d + std::hypot(d, ak_matrix(1,0));
  }
}

template <class DataType, class Derived>
std::enable_if_t<IsComplex<DataType>(), DataType>
WilkinsonShift(const Eigen::MatrixBase<Derived> &ak_matrix, const bool a_first = true) {
    DataType trace = ak_matrix.trace();
    DataType tmp = std::sqrt(trace * trace - 4.0 * ak_matrix.determinant());
    DataType ev1 = (trace + tmp) / 2.0;
    DataType ev2 = (trace - tmp) / 2.0;

    DataType entry;
    if (a_first) {
      entry = ak_matrix(1, 1);
    } else {
      entry = ak_matrix(0, 0);
    }
      if (std::abs(ev1 - entry) < std::abs(ev2 - entry)) {        // return the nearest eigenvalue
        return ev1;
      } else {
        return ev2;
      }
}



/* Applies a single givens rotation to a tridiagonal Matrix
 * Parameter:
 *  - a_matrix: Triagonal Matrix
 *  - ak_k:     Index of upper row in the rotation
 *  - ak_c:       Parameter 'c' in the givens rotation
 *  - ak_s:       Parameter 's' in the givens rotation
 */
template <class DataType, bool is_symmetric, bool is_first, bool is_last,
         class Derived>
std::enable_if_t<is_symmetric && std::is_arithmetic<DataType>::value, void>
ApplyGivens(const Eigen::MatrixBase<Derived> &a_matrix, const int ak_k,
            const DataType ak_c, const DataType ak_s) {
  Eigen::MatrixBase<Derived>& matrix =
    const_cast<Eigen::MatrixBase<Derived>&>(a_matrix);
  Eigen::Matrix<typename Derived::Scalar, 2, 2> Q;
  Q(0, 0) = ak_c;
  Q(1, 1) = ak_c;
  Q(0, 1) = -ak_s;
  Q(1, 0) = ak_s;

  // Previous Column
  if constexpr (!is_first) {
    matrix(Eigen::seq(ak_k, ak_k + 1), ak_k -1) = Q.adjoint() * matrix(Eigen::seq(ak_k, ak_k + 1), ak_k -1);
    matrix(ak_k -1, Eigen::seq(ak_k, ak_k + 1)) = matrix(ak_k - 1, Eigen::seq(ak_k, ak_k + 1)) * Q;
//    matrix(ak_k -1, ak_k + 1) = 0;
//    matrix(ak_k + 1, ak_k -1) = 0;
  }

  // Center Block
  matrix(Eigen::seq(ak_k, ak_k + 1), Eigen::seq(ak_k, ak_k + 1)) = Q.adjoint() * matrix(Eigen::seq(ak_k, ak_k +1), Eigen::seq(ak_k, ak_k +1));
  matrix(Eigen::seq(ak_k, ak_k + 1), Eigen::seq(ak_k, ak_k + 1)) = matrix(Eigen::seq(ak_k, ak_k +1), Eigen::seq(ak_k, ak_k +1)) * Q;

  // Next Column
  if constexpr (!is_last) {
    matrix(Eigen::seq(ak_k, ak_k + 1), ak_k + 2) = Q.adjoint() * matrix(Eigen::seq(ak_k, ak_k + 1), ak_k + 2);
    matrix(ak_k + 2, Eigen::seq(ak_k, ak_k + 1)) = matrix(ak_k + 2, Eigen::seq(ak_k, ak_k + 1)) * Q;
  }
  return;
}

template <class DataType,  bool is_symmetric, class Derived>
//std::enable_if_t<!is_symmetric, void>
void
ApplyGivens(const Eigen::MatrixBase<Derived> &a_matrix, const int ak_k,
            const DataType ak_c, const DataType ak_s,
            const DataType ak_sconj) {
  Eigen::MatrixBase<Derived>& matrix = const_cast<typename Eigen::MatrixBase<
    Derived>&>(a_matrix);
  Eigen::Matrix<DataType, 2, 2> Q;
  Q(0, 0) = ak_c;
  Q(1, 1) = ak_c;
  Q(0, 1) = -ak_s;
  Q(1, 0) = ak_sconj;
  int start = std::max(0, ak_k -1);
  long end = std::min(long{ak_k + 2}, long{matrix.rows() - 1});
  matrix(Eigen::seq(ak_k, ak_k+1), Eigen::seq(start, matrix.rows() -1)) =
    Q.adjoint() * matrix(Eigen::seq(ak_k, ak_k+1),
        Eigen::seq(start, matrix.rows() -1));
  matrix(Eigen::seq(0, end), Eigen::seq(ak_k, ak_k+1)) =
    matrix(Eigen::seq(0, end), Eigen::seq(ak_k, ak_k+1)) * Q;
  return;
}


/* Calculate the entries of a Givens Matrix
 * Parameter:
 * - ak_a: first entry
 * - ak_b: entry to eliminate
 * Return: Vector containing {c, s, std::conj(s)}
 */
template<class DataType> inline
std::enable_if_t<std::is_arithmetic<DataType>::value, std::vector<DataType>>
GetGivensEntries(const DataType& ak_a, const DataType& ak_b) {
  std::vector<DataType> res(2);
  DataType r = std::hypot(ak_a, ak_b);
  res.at(0) = std::abs(ak_a) / r;
  res.at(1) = ak_b / r * std::copysign(1, ak_a);
  return res;
}

template<class DataType> inline
std::enable_if_t<IsComplex<DataType>(), std::vector<DataType>>
GetGivensEntries(const DataType& ak_a, const DataType& ak_b) {
  typedef typename DataType::value_type real;
  std::vector<DataType> res(3);
  real absa = std::abs(ak_a);
  real absb = std::abs(ak_b);
  real r = std::hypot(absa, absb);
  res.at(0) = absa / r;
  res.at(1) = std::polar(std::abs(ak_b) / r, std::arg(ak_a) - std::arg(ak_b));
  res.at(2) = std::conj(res.at(1));
  return res;
}


/* Executes one step of the implicit qr algorithm for a tridiagonal Matrix
 * Parameter:
 * - a_matrix: Tridiagonal Matrix
 * Return: void
 */
template <class DataType, bool is_symmetric, typename Derived>
void
ImplicitQrStep(const Eigen::MatrixBase<Derived> &a_matrix,
               const double = 1e-12) {
  DataType shift = WilkinsonShift<DataType>(a_matrix(Eigen::lastN(2),
        Eigen::lastN(2)));
  int n = a_matrix.rows();
  auto entries = GetGivensEntries<>(a_matrix(0, 0) - shift, a_matrix(1, 0));
  if constexpr (is_symmetric) {
    switch (n) {                                                                  // Initial step
      case 2:
        ApplyGivens<DataType, is_symmetric, true, true>(a_matrix, 0,
            entries.at(0), entries.at(1));
        return;
      case 3:
        ApplyGivens<DataType, is_symmetric, true, false>(a_matrix, 0,
            entries.at(0), entries.at(1));
        entries = GetGivensEntries<>(a_matrix(1, 0), a_matrix(2,0));
        ApplyGivens<DataType, is_symmetric, false, true>(a_matrix, 1,
            entries.at(0), entries.at(1));
        return;
      default:
        ApplyGivens<DataType, is_symmetric, true, false>(a_matrix, 0,
            entries.at(0), entries.at(1));
    }
    for (int k = 1; k < n - 2; ++k) {                                             // Buldge Chasing
      entries = GetGivensEntries<>(a_matrix(k, k-1), a_matrix(k+1, k-1));
//      if( d(gen)) {
//       const_cast<typename Derived::Scalar&>(a_matrix(k + 2, k + 1)) *= -1;
//       const_cast<typename Derived::Scalar&>(a_matrix(k + 1, k + 2)) *= -1;
//      }
      ApplyGivens<DataType, is_symmetric, false, false>(a_matrix, k,
          entries.at(0), entries.at(1));
    }
    entries = GetGivensEntries<>(a_matrix(n-2, n-3), a_matrix(n-1, n-3));
    ApplyGivens<DataType, is_symmetric, false, true>(a_matrix, n-2,
        entries.at(0), entries.at(1));
  } else {
    // TODO improve readability
    if (entries.size() == 2) entries.push_back(entries.at(1));
    ApplyGivens<DataType, is_symmetric>(a_matrix, 0, entries.at(0),
        entries.at(1), entries.at(2));
    for (int k = 1; k < n - 1; ++k) {                                             // Buldge Chasing
      entries = GetGivensEntries<>(a_matrix(k, k-1), a_matrix(k+1, k-1));
      if (entries.size() == 2) entries.push_back(entries.at(1));
      ApplyGivens<DataType, is_symmetric>(a_matrix, k, entries.at(0),
          entries.at(1), entries.at(2));
      // TODO include this into the apply givens?
      const_cast<Eigen::MatrixBase<Derived>&>(a_matrix)(k + 1, k - 1) = 0.0;
    }
  }
//  std::cout << a_matrix.diagonal(-1) << std::endl
//            << "diag: " << std::endl
//            << a_matrix.diagonal() << std::endl
//            << "superdiag: " << std::endl
//            << a_matrix.diagonal(1) << std::endl
//            << std::endl << std::endl;
  return;
}

template <class DataType, bool is_symmetric, typename Derived>
void
ReverseImplicitQrStep(const Eigen::MatrixBase<Derived> &a_matrix,
               const double = 1e-12) {
  DataType shift = WilkinsonShift<DataType>(a_matrix(Eigen::seqN(0, 2),
        Eigen::seqN(0, 2)), false);
  int n = a_matrix.rows();
  //std::cout << "shift: " << shift << std::endl;
  auto entries = GetGivensEntries<>(a_matrix(n-1, n-1) - shift, a_matrix(n-1, n-2));
  if constexpr (is_symmetric) {
//  if constexpr (false) {
    switch (n) {                                                                  // Initial step
      case 2:
        ApplyGivens<DataType, is_symmetric, true, true>(a_matrix, 0,
            entries.at(0), -entries.at(1));
        return;
      case 3:
        ApplyGivens<DataType, is_symmetric, false, true>(a_matrix, 1,
            entries.at(0), -entries.at(1));
        entries = GetGivensEntries<>(a_matrix(2, 1), a_matrix(2,0));
        ApplyGivens<DataType, is_symmetric, true, false>(a_matrix, 0,
            entries.at(0), -entries.at(1));
        return;
      default:
        ApplyGivens<DataType, is_symmetric, false, true>(a_matrix, n - 2,
            entries.at(0), -entries.at(1));
    }
    for (int k = n - 2 ; k >= 2; --k) {                                             // Buldge Chasing
      entries = GetGivensEntries<>(a_matrix(k + 1, k), a_matrix(k + 1, k - 1));
//      if( d(gen)) {
//       const_cast<typename Derived::Scalar&>(a_matrix(k + 2, k + 1)) *= -1;
//       const_cast<typename Derived::Scalar&>(a_matrix(k + 1, k + 2)) *= -1;
//      }
      ApplyGivens<DataType, is_symmetric, false, false>(a_matrix, k - 1,
          entries.at(0), -entries.at(1));
    }
    entries = GetGivensEntries<>(a_matrix(2, 1), a_matrix(2, 0));
    ApplyGivens<DataType, is_symmetric, true, false>(a_matrix, 0,
        entries.at(0), -entries.at(1));
  } else {
//    std::cout << "Non symmetric solver" << std::endl;
    // TODO improve readability
    if (entries.size() == 2) entries.push_back(entries.at(1));
    // Check Complex Version
    ApplyGivens<DataType, is_symmetric>(a_matrix, n-2, entries.at(0),
        -entries.at(1), -entries.at(2));
    for (int k = n - 2; k >= 1 ; --k) {                                             // Buldge Chasing
      entries = GetGivensEntries<>(a_matrix(k + 1, k), a_matrix(k + 1, k - 1));
      if (entries.size() == 2) entries.push_back(entries.at(1));
      ApplyGivens<DataType, is_symmetric>(a_matrix, k-1, entries.at(0),
          -entries.at(1), -entries.at(2));
      // TODO include this into the apply givens?
      const_cast<Eigen::MatrixBase<Derived>&>(a_matrix)(k + 1, k - 1) = 0.0;
    }
  }
  //std::cout << a_matrix << std::endl << std::endl;
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
DoubleShiftParameter(const Eigen::MatrixBase<Derived> &ak_matrix, const int a_index = 1) {
  typedef typename Derived::Scalar DataType;
  std::vector<typename Derived::Scalar> res(2);
  //  If Real use the same but with the eigenvalues
  res.at(0) = -ak_matrix.trace();
  res.at(1) = ak_matrix.determinant();
  // TODO implicit shift when possible?
  if (res.at(0) * res.at(0) > 4.0 * res.at(1)) {
    //return std::vector<typename Derived::Scalar>{}; // TODO remove this test
    DataType tmp = std::sqrt(res.at(0) * res.at(0) - 4.0 * res.at(1));
    DataType ev1 = (-res.at(0) + tmp) / 2.0;
    DataType ev2 = (-res.at(0) - tmp) / 2.0;
    if (std::abs(ev1 - ak_matrix(a_index, a_index)) < std::abs(ev2 - ak_matrix(a_index, a_index))) {
      res.at(0) = -2.0 * ev1;
      res.at(1) = ev1 * ev1;
    } else {
      res.at(0) = -2.0 * ev2;
      res.at(1) = ev2 * ev2;
    }
  }
  return res;
}

/* Executes one step of the double shift algorithm
 * Parameter:
 * - a_matrix: Tridiagonal Matrix
 * Return: void
 */
template <class Derived>
void DoubleShiftQrStep(const Eigen::MatrixBase<Derived> &a_matrix,
                       const double ak_tol = 1e-12) {
  typedef Eigen::MatrixBase<Derived> MatrixType;
  typedef Eigen::Matrix<typename Derived::Scalar, -1, -1> Matrix;
  MatrixType& matrix = const_cast<MatrixType &>(a_matrix);
  int n = a_matrix.rows();
  std::vector<typename Derived::Scalar> shift =
      DoubleShiftParameter<>(a_matrix(Eigen::lastN(2), Eigen::lastN(2)));
//  if (shift.size() == 0) {    // TODO: remove this test
//    ImplicitQrStep<typename Derived::Scalar, false>(a_matrix, ak_tol);
//    return;
//  }
  // Only first three entries of first col needed
  Matrix matrix2 = matrix;
  Matrix m1 = a_matrix(Eigen::seqN(0,3), Eigen::all) *
    a_matrix(Eigen::all, 0) + shift.at(0) *
    a_matrix(Eigen::seqN(0,3), 0) + shift.at(1) * Matrix::Identity(3, 1);
  Eigen::Vector<typename Derived::Scalar, -1> w = GetHouseholderVector(m1, ak_tol);
  long end = std::min(4, n);
  ApplyHouseholderRight(w, matrix(Eigen::seqN(0, end), Eigen::seqN(0, 3)));
  ApplyHouseholderLeft(w, matrix(Eigen::seqN(0, 3), Eigen::all));
  for (int i = 0; i < n - 3; ++i) {
    Eigen::Vector<typename Derived::Scalar, -1> w = GetHouseholderVector(matrix(
          Eigen::seqN(i + 1, 3), i), ak_tol);
    end = std::min(i+4, n-1);
    ApplyHouseholderRight(w, matrix(Eigen::seq(0, end), Eigen::seqN(i+1, 3)));
    ApplyHouseholderLeft(w, matrix(Eigen::seqN(i+1, 3), Eigen::seq(i, n-1)));
    matrix(Eigen::seqN(i + 2, 2), i) = Matrix::Zero(2, 1);                      // Set Round off errors to 0
  }
  // Maybe Givens?
  w = GetHouseholderVector(matrix(Eigen::lastN(2), n-3), ak_tol);
  ApplyHouseholderRight(w, matrix(Eigen::all, Eigen::lastN(2)));
  ApplyHouseholderLeft(w, matrix(Eigen::lastN(2), Eigen::lastN(3)));
}

template <class Derived>
void ReverseDoubleShiftQrStep(const Eigen::MatrixBase<Derived> &a_matrix,
                       const double ak_tol = 1e-12) {
  typedef Eigen::MatrixBase<Derived> MatrixType;
  typedef Eigen::Matrix<typename Derived::Scalar, -1, -1> Matrix;
  MatrixType& matrix = const_cast<MatrixType &>(a_matrix);
  int n = a_matrix.rows();
  std::vector<typename Derived::Scalar> shift =
      DoubleShiftParameter<>(a_matrix(Eigen::seqN(0, 2), Eigen::seqN(0, 2)), 0);
//  if (shift.size() == 0) {    // TODO: remove this test
//    ImplicitQrStep<typename Derived::Scalar, false>(a_matrix, ak_tol);
//    return;
//  }
  // Only last three entries of last row needed
  Matrix m1 = a_matrix(n - 1, Eigen::all) * a_matrix(Eigen::all,
      Eigen::lastN(3)) + shift.at(0) * a_matrix(n - 1, Eigen::lastN(3));
  m1(0, 2) += shift.at(1);
  ApplyReverseHouseholder<>(m1.transpose(), matrix, n - 3, n, ak_tol);                                  // Calc initial Step
  for (int i = n - 1; i > 2; --i) {
    ApplyReverseHouseholder<>(matrix(i, Eigen::seqN(i - 3, 3)).transpose(),
        matrix, i - 3, i + 1, ak_tol);          // Buldge Chasing
    matrix(i, Eigen::seqN(i - 3, 2)) = Matrix::Zero(1, 2);                      // Set Round off errors to 0
  }
  // Maybe Givens?
  ApplyReverseHouseholder(matrix(2, Eigen::seqN(0, 2)), matrix, 0, 3, ak_tol);
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
bool DeflateDiagonal(const Eigen::MatrixBase<Derived> &a_matrix, int &a_begin,
                      int &a_end, const double ak_tol = 1e-12) {
  bool state = true;
  for (int i = a_end; i > a_begin; --i) {
    if (std::abs(a_matrix(i, i - 1)) < ak_tol * (std::abs(a_matrix(i, i)) +
          std::abs(a_matrix(i - 1, i - 1)))) {
      const_cast<Eigen::MatrixBase<Derived> &>(a_matrix)(i, i - 1) = 0;
      if (!state) {
        a_begin = i;
        return false;                                                             // Subblock to solve found
      }
    } else if (state) {                                                           // Start of the block found
      a_end = i;
      state = false;
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
bool DeflateSchur(const Eigen::MatrixBase<Derived> &a_matrix, int &a_begin,
                   int &a_end, const double ak_tol = 1e-12) {
  bool state = true;
  for (int i = a_end; i > a_begin; --i) {
    //if (std::abs(a_matrix(i, i - 1)) < ak_tol * std::abs(a_matrix(i, i) +
    if (std::abs(a_matrix(i, i - 1)) < 1e-7 * (std::abs(a_matrix(i, i)) +
          std::abs(a_matrix(i - 1, i - 1)))) {
      const_cast<Eigen::MatrixBase<Derived> &>(a_matrix)(i, i - 1) = 0;
      if (!state) {
        a_begin = i;
        return false;                                                             // Subblock to solve found
      }
    } else if (state && (i - 1 > a_begin) &&
               (std::abs(a_matrix(i - 1, i - 2)) >= ak_tol * (std::abs(a_matrix(
                  i - 2, i - 2)) + std::abs(a_matrix(i - 1, i - 1))))) {          // Start of the block found
      a_end = i;
      --i;                                                                        // Next index already checked
      state = false;
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
CalcEigenvaluesFromSchur(const Eigen::MatrixBase<Derived>& ak_matrix,
                         const bool ak_matrix_is_diagonal = false) {
  std::vector<DataType> res(ak_matrix.rows());
  if (ak_matrix_is_diagonal || ak_matrix.rows() == 1) {
    for (int i = 0; i < ak_matrix.rows(); ++i) {
      res.at(i) = ak_matrix(i, i);
    }
  } else {
    for (int i = 0; i < ak_matrix.rows() - 1; ++i) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-equal"
      if (std::abs(ak_matrix(i + 1, i)) == 0) {                                   // Eigenvalue in diagonal block
#pragma GCC diagnostic pop
        res.at(i) = ak_matrix(i, i);
      } else {                                                                    // Eigenvalue in a 2x2 block
        typename Derived::Scalar d = (ak_matrix(i, i) +
            ak_matrix(i + 1, i + 1)) / 2.0;
        DataType pq = std::sqrt(DataType{ d * d - ak_matrix(i, i) *
            ak_matrix(i + 1, i + 1) + ak_matrix(i, i + 1) *
            ak_matrix(i + 1, i)});
        res.at(i) = d - pq;                                                       // First eigenvalue
        ++i;
        res.at(i) = d + pq;                                                       // Second eigenvalue
      }
    }
#pragma GCC diagnostic ignored "-Wfloat-equal"
#pragma GCC diagnostic push
    if (std::abs(ak_matrix(ak_matrix.rows() - 1, ak_matrix.rows() - 2)) == 0.0)   // Last EV is in a diagonal block
#pragma GCC diagnostic pop
      res.at(ak_matrix.rows() - 1) = ak_matrix(ak_matrix.rows() - 1,
          ak_matrix.rows() - 1);                                                  // Add the last eigenvalue
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
  typedef Eigen::MatrixBase<Derived> MatrixType;
  typedef Eigen::Block<Derived, -1, -1, false> StepMatrix;
  int begin = 0;
  int end = a_matrix.rows() - 1;
  int end_of_while = 0;
  bool tridiagonal_result = true;
  std::vector<std::complex<double>> res;
  void (*step)(const Eigen::MatrixBase<StepMatrix> &, const double);
  bool (*deflate)(const Eigen::MatrixBase<Derived> &, int &, int &,
                  const double);
  if constexpr (std::is_arithmetic<typename Derived::Scalar>::value &&
      !ak_is_hermitian) {
      end_of_while = 1;
      step = &DoubleShiftQrStep<StepMatrix>;
//      step = &ReverseDoubleShiftQrStep<StepMatrix>;
      deflate = &DeflateSchur<Derived>;
      tridiagonal_result = false;
  } else {
    step = &ImplicitQrStep<typename MatrixType::Scalar, ak_is_hermitian,
//    step = &ReverseImplicitQrStep<typename MatrixType::Scalar, ak_is_hermitian,
         StepMatrix>;
    deflate = &DeflateDiagonal<Derived>;
  }
  while (end_of_while < end) {
    if (deflate(a_matrix, begin, end, ak_tol)) {
      end = begin - 1;
      begin = 0;
    } else {
      step(const_cast<MatrixType&>(a_matrix)(Eigen::seq(begin, end),
              Eigen::seq(begin, end)), ak_tol);
    }
  }
  // Or form Diagonal?
  return CalcEigenvaluesFromSchur<DataType>(a_matrix, tridiagonal_result);
}

/* Calculate the eigenvalues of a Matrix using the QR decomposition
 * Parameter:
 * - a_matrix: Square Matrix
 * - ak_tol: Tolerance for considering a value 0
 * Return: Unordered Vector of (complex) eigenvalues
 */
template <bool IsHermitian, typename Derived, class DataType = double>
typename std::enable_if_t<std::is_arithmetic<typename Derived::Scalar>::value,
          std::vector<std::complex<DataType>>>
QrMethod(const Eigen::MatrixBase<Derived> &ak_matrix,
         const double ak_tol = 1e-12) {
  assert(ak_matrix.rows() == ak_matrix.cols());
  static_assert(std::is_convertible<typename Derived::Scalar, DataType>::value,
      "Matrix Elements must be convertible to DataType");
  typedef Eigen::Matrix<DataType, -1, -1> Matrix;

  Matrix A = ak_matrix;
//  const bool k_is_symmetric = IsHermitian(ak_matrix, ak_tol);
//  HessenbergTransformation<>(A, ak_tol, k_is_symmetric);
//  if (k_is_symmetric) {                                                           // Necessary because it is a template parameter
//    return QrIterationHessenberg<std::complex<DataType>, true>(A, ak_tol);
//  } else {
//    return QrIterationHessenberg<std::complex<DataType>, false>(A, ak_tol);
//  }
    return QrIterationHessenberg<std::complex<DataType>, IsHermitian>(A, ak_tol);
}


template <bool IsHermitian, typename Derived, class DataType = std::complex<double>>
typename std::enable_if_t<IsComplex<typename Derived::Scalar>(),
  std::vector<DataType>> QrMethod(const Eigen::MatrixBase<Derived> &ak_matrix,
                                  const double ak_tol = 1e-12) {
  assert(ak_matrix.rows() == ak_matrix.cols());
  static_assert(std::is_convertible<typename Derived::Scalar, DataType>::value,
      "Matrix Elements must be convertible to DataType");
  typedef Eigen::Matrix<DataType, -1, -1> Matrix;

  Matrix A = ak_matrix;
//  const bool k_is_hermitian = IsHermitian(ak_matrix, ak_tol);
//  HessenbergTransformation<>(A, ak_tol, k_is_hermitian);

//  if (k_is_hermitian) {                                                           // Necessary because it is a template parameter
//    return QrIterationHessenberg<DataType, true>(A.real(), ak_tol);
//  } else {
//    return QrIterationHessenberg<DataType, false>(A, ak_tol);
//  }
  return QrIterationHessenberg<DataType, IsHermitian>(A, ak_tol);
}

} // namespace nla_exam
#endif
