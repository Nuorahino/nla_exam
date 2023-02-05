#ifndef QR_HH_
#define QR_HH_

/*
 * TODO: optimize for eigen (noalias)
 */

#include <cmath>
#include <complex>
#include <type_traits>
#include <vector>
#include <iostream>
#include <cassert>

#include <eigen3/Eigen/Dense>

#include "helpfunctions/helpfunctions.hh"

namespace nla_exam {
/* Create a Householder Reflection
 * Parameter:
 * - ak_x: Vector determining the reflection
 * - a_p: Returns the Householder Matrix
 * Return: void
 */
template <class Derived, class Derived2> inline
void
CreateHouseholder(const Eigen::MatrixBase<Derived> &ak_x,
                  const Eigen::MatrixBase<Derived2> &a_p,
                  const double ak_tol = 1e-14) {
  typedef typename Eigen::Matrix<typename Derived::Scalar, -1, -1> MatrixType;
  Eigen::MatrixBase<Derived2>& p = const_cast<typename Eigen::MatrixBase
    <Derived2>&>(a_p);

  // TODO improve break criteria
  //if( ak_x.squaredNorm() <= ak_tol ) return;
  if (ak_x.squaredNorm() <= ak_tol * ak_tol) return;                             // Avoid devision by 0
  MatrixType u = ak_x;
  typename Derived::Scalar alpha = u.norm();
  if constexpr (IsComplex<typename Derived::Scalar>()) {
    alpha *= std::polar(1.0, arg(u(0)));                                          // Choise to avoid loss of significance
  } else {
    if (u(0) < 0) alpha *= -1;
  }
  u(0) = u(0) + alpha;
  p.setIdentity();                                                                // Leave rest unchanged
  p(Eigen::lastN(u.rows()), Eigen::lastN(u.rows())) -=
                      2 * u * u.adjoint() / u.squaredNorm();                      // Calculate the relevant block
  return;
}

/* Transforms a Matrix to Hessenberg form
 * Parameter:
 * - a_matrix: Matrix to transform
 * Return: The unitary matrix used in the similarity trasnformation
 */
template <class Derived>
Eigen::Matrix<typename Derived::Scalar, -1, -1>
HessenbergTransformation(const Eigen::MatrixBase<Derived> &a_matrix,
                         const double ak_tol = 1e-14,
                         const bool a_is_hermitian = false) {
  typedef Eigen::Matrix<typename Derived::Scalar, -1, -1> Matrix;
  typedef Eigen::MatrixBase<Derived> MatrixType;
  MatrixType& matrix = const_cast<MatrixType&>(a_matrix);

  Matrix q = Matrix::Identity(a_matrix.rows(), a_matrix.cols());                  // q is transformation Matrix
  Matrix p(a_matrix.rows(), a_matrix.rows());                                     // p is Householder reflection
  for (int i = 0; i < matrix.rows() - 1; ++i) {
    CreateHouseholder<>(matrix(Eigen::lastN(a_matrix.rows() - i - 1), i), p,
        ak_tol);                                                                  // Calc Householder Matrix
    matrix = p.adjoint() * matrix * p;                                            // Transformation Step
    matrix(Eigen::lastN(matrix.rows() - i - 2), i) =
      Matrix::Zero(matrix.rows() - i - 2, 1);                                     // Set Round off errors to 0
    if (a_is_hermitian) {
      matrix(i, Eigen::lastN(a_matrix.rows() - i - 2))
        = Matrix::Zero(1, a_matrix.rows() - i - 2);                               // Set Round off errors to 0
    }
    q *= p;                                                                       // Build the transformation Matrix
  }
  if constexpr (IsComplex<typename Derived::Scalar>()) {
    if (a_is_hermitian) {                                                         // Transform complex Hermitian Matrix to Real
      for(int i = 1; i < a_matrix.rows(); ++i) {
            matrix(i-1, i) = std::abs(a_matrix(i-1, i));
            matrix(i, i-1) = std::abs(a_matrix(i, i-1));
      }
    }
  }
  return q;
}


/* Get Wilkinson shift parameter for a given 2x2 Matrix
 * Parameter:
 * - ak_matrix: 2x2 Matrix of which to calculate the shift parameter
 * Return: value of the shift parameter
 */
template <class DataType, class Derived> inline
std::enable_if_t<std::is_arithmetic<DataType>::value, DataType>
WilkinsonShift(const Eigen::MatrixBase<Derived> &ak_matrix) {
  DataType d = (ak_matrix(0, 0) - ak_matrix(1, 1)) / 2.0;
  if (d >= 0) {
    return ak_matrix(1, 1) + d - std::sqrt(d * d + ak_matrix(1, 0) *
        ak_matrix(1, 0));
  } else {
    return ak_matrix(1, 1) + d + std::sqrt(d * d + ak_matrix(1, 0) *
        ak_matrix(1, 0));
  }
}

template <class DataType, class Derived>
std::enable_if_t<IsComplex<DataType>(), DataType>
WilkinsonShift(const Eigen::MatrixBase<Derived> &ak_matrix) {
    DataType trace = ak_matrix.trace();
    DataType tmp = std::sqrt(trace * trace - 4.0 * ak_matrix.determinant());
    DataType ev1 = (trace + tmp) / 2.0;
    DataType ev2 = (trace - tmp) / 2.0;
  if (std::abs(ev1 - ak_matrix(1, 1)) < std::abs(ev2 - ak_matrix(1, 1))) {        // return the nearest eigenvalue
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
 *  - a_buldge:  Current Value of the buldge
 */
template <class DataType, bool is_symmetric, class Derived>
std::enable_if_t<is_symmetric && std::is_arithmetic<DataType>::value, void>
ApplyGivens(const Eigen::MatrixBase<Derived> &a_matrix, const int ak_k,
             const DataType ak_c, const DataType ak_s, DataType &a_buldge) {
  Eigen::MatrixBase<Derived>& matrix =
    const_cast<Eigen::MatrixBase<Derived>&>(a_matrix);
  DataType alpha;
  DataType beta;

  // Previous Column
  if (ak_k > 0) {
    alpha = matrix(ak_k, ak_k - 1);
    beta = a_buldge;
    matrix(ak_k, ak_k - 1) = (ak_c * alpha) + (ak_s * beta);
    a_buldge = -ak_s * alpha + ak_c * beta;
  }

  // Center Block
  alpha = matrix(ak_k, ak_k);
  DataType alpha_2 = matrix(ak_k + 1, ak_k + 1);
  beta = matrix(ak_k + 1, ak_k);
  matrix(ak_k, ak_k) = ak_c * ak_c * alpha + ak_s * ak_s * alpha_2 +
    2.0 * ak_c * ak_s * beta;
  matrix(ak_k + 1, ak_k) = -ak_s * (ak_c * alpha + ak_s * beta) +
    ak_c * (ak_c * beta + ak_s * alpha_2);
  matrix(ak_k + 1, ak_k + 1) = ak_c * ak_c * alpha_2 + ak_s * ak_s * alpha -
    2.0 * ak_c * ak_s * beta;

  // Next Column
  if (ak_k < matrix.rows() - 2) {
    alpha = 0;                                                                    // new buldge
    beta = matrix(ak_k + 2, ak_k + 1);
    a_buldge = ak_c * alpha + ak_s * beta;
    matrix(ak_k + 2, ak_k + 1) = -ak_s * alpha + ak_c * beta;
  }
  return;
}

template <class DataType,  bool is_symmetric, class Derived>
std::enable_if_t<!is_symmetric, void>
ApplyGivens(const Eigen::MatrixBase<Derived> &a_matrix, const int ak_k,
            const DataType ak_c, const DataType ak_s,
            const DataType ak_sconj) {
  typedef Eigen::Matrix<DataType, -1, -1> Matrix;
  Eigen::MatrixBase<Derived>& matrix = const_cast<typename Eigen::MatrixBase<
    Derived>&>(a_matrix);

  Matrix Q = Matrix::Identity(2, 2);
  Q(0, 0) = ak_c;
  Q(1, 1) = ak_c;
  Q(0, 1) = ak_s;
  Q(1, 0) = -ak_sconj;
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
  res.at(0) = ak_a / r;
  res.at(1) = ak_b / r;
  return res;
}

template<class DataType> inline
std::enable_if_t<IsComplex<DataType>(), std::vector<DataType>>
GetGivensEntries(const DataType& ak_a, const DataType& ak_b) {
  assert(1 != 0);
  typedef typename DataType::value_type real;
  std::vector<DataType> res(3);
  real absa = std::abs(ak_a);
  real absb = std::abs(ak_b);
  real r = std::hypot(absa, absb);
  res.at(0) = absa / r;
  res.at(1) = -ak_a / absa * (std::conj(ak_b) / r);
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
               const double = 1e-14) {
  DataType shift = WilkinsonShift<DataType>(a_matrix(Eigen::lastN(2),
        Eigen::lastN(2)));
  DataType buldge = 0;
  auto entries = GetGivensEntries<>(a_matrix(0, 0) - shift, a_matrix(1, 0));
  if constexpr (is_symmetric) {                                                   // Initial step
    ApplyGivens<DataType, is_symmetric>(a_matrix, 0, entries.at(0),
        entries.at(1), buldge);
  } else {
    ApplyGivens<DataType, is_symmetric>(a_matrix, 0, entries.at(0),
        entries.at(1), entries.at(2));
  }

  for (int k = 1; k < a_matrix.rows() - 1; ++k) {                                 // Buldge Chasing
    if constexpr (is_symmetric) {
      entries = GetGivensEntries<>(a_matrix(k, k-1), buldge);
      ApplyGivens<DataType, is_symmetric>(a_matrix, k, entries.at(0),
      entries.at(1), buldge);
    } else {
      entries = GetGivensEntries<>(a_matrix(k, k-1), a_matrix(k+1, k-1));
      ApplyGivens<DataType, is_symmetric>(a_matrix, k, entries.at(0),
          entries.at(1), entries.at(2));
    }
  //TODO  Maybe neccesary
//    if (std::abs(buldge) < 1e-14)
//      break;
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
  typedef typename Derived::Scalar DataType;
  std::vector<typename Eigen::MatrixBase<Derived>::Scalar> res(2);
  //  If Real use the same but with the eigenvalues
  res.at(0) = -ak_matrix.trace();
  res.at(1) = ak_matrix.determinant();
  // TODO implicit shift when possible?
  if (res.at(0) * res.at(0) > 4.0 * res.at(1)) {
    DataType tmp = std::sqrt(res.at(0) * res.at(0) - 4.0 * res.at(1));
    DataType ev1 = (-res.at(0) + tmp) / 2.0;
    DataType ev2 = (-res.at(0) - tmp) / 2.0;
    if (std::abs(ev1 - ak_matrix(1,1)) < std::abs(ev2 - ak_matrix(1,1))) {
      res.at(0) = -2.0 * ev1;
      res.at(1) = ev1 * ev1;
    } else {
      res.at(0) = -2.0 * ev2;
      res.at(1) = ev2 * ev2;
    }
  }
  return res;
}

// TODO Optimize using the known Matrix structure
/* Executes one step of the double shift algorithm
 * Parameter:
 * - a_matrix: Tridiagonal Matrix
 * Return: void
 */
template <class Derived>
void DoubleShiftQrStep(const Eigen::MatrixBase<Derived> &a_matrix,
                       const double ak_tol = 1e-14) {
  typedef Eigen::MatrixBase<Derived> MatrixType;
  typedef Eigen::Matrix<typename Derived::Scalar, -1, -1> Matrix;
  int n = a_matrix.rows();
  std::vector<typename Derived::Scalar> shift =
      DoubleShiftParameter<>(a_matrix(Eigen::lastN(2), Eigen::lastN(2)));
  Matrix m1 = a_matrix * a_matrix(Eigen::all, 0) + shift.at(0) *
    a_matrix(Eigen::all, 0) + shift.at(1) * Matrix::Identity(n, 1);               // Only compute the first col
  Matrix p(n,n);                                                                  // Householder Matrix
  //CreateHouseholder<>(m1, p, ak_tol);                                               // Calc initial Step
  CreateHouseholder<>(m1, p, 1e-14);                                               // Calc initial Step
  const_cast<MatrixType &>(a_matrix) = p.adjoint() * a_matrix * p;                // Transformation Step
  for (int i = 0; i < n - 2; ++i) {
    //CreateHouseholder<>(a_matrix(Eigen::seq(i + 1, n - 1), i), p, ak_tol);          // Buldge Chasing
    CreateHouseholder<>(a_matrix(Eigen::seq(i + 1, n - 1), i), p, 1e-14);          // Buldge Chasing
    const_cast<MatrixType&>(a_matrix) = p.adjoint() * a_matrix * p;               // Transformation Step
    const_cast<MatrixType&>(a_matrix)( Eigen::seq(i + 2, n - 1), i) =
                    Matrix::Zero(n - i - 2, 1);                                   // Set Round off errors to 0
  }
}


// TODO Eventually return threads
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
                      int &a_end, const double ak_tol = 1e-14) {
  bool state = true;
  for (int i = a_end; i > a_begin; --i) {
    if (std::abs(a_matrix(i, i - 1)) < ak_tol * std::abs(a_matrix(i, i) +
          a_matrix(i - 1, i - 1))) {
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
                   int &a_end, const double ak_tol = 1e-14) {
  bool state = true;
  for (int i = a_end; i > a_begin; --i) {
    if (std::abs(a_matrix(i, i - 1)) < ak_tol * std::abs(a_matrix(i, i) +
          a_matrix(i - 1, i - 1))) {
      const_cast<Eigen::MatrixBase<Derived> &>(a_matrix)(i, i - 1) = 0;
      if (!state) {
        a_begin = i;
        return false;                                                             // Subblock to solve found
      }
    } else if (state && (i - 1 > a_begin) &&
               (std::abs(a_matrix(i - 1, i - 2)) >= ak_tol *
                std::abs(a_matrix(i - 2, i - 2) + a_matrix(i - 1, i - 1)))) {     // Start of the block found
      a_end = i;
      --i;                                                                        // Next index already checked
      state = false;
    }
  }
  return state;
}


// TODO Needs changes for complex EVs
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
  if (ak_matrix_is_diagonal) {
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
                        const double ak_tol = 1e-14) {
  typedef Eigen::MatrixBase<Derived> MatrixType;
  typedef Eigen::Block<Derived, -1, -1, false> StepMatrix;
  int begin = 0;
  int end = a_matrix.rows() - 1;
  int end_of_while = 0;
  std::vector<std::complex<double>> res;
  void (*step)(const Eigen::MatrixBase<StepMatrix> &, const double);
  bool (*deflate)(const Eigen::MatrixBase<Derived> &, int &, int &,
                  const double);

  if constexpr (std::is_arithmetic<typename Derived::Scalar>::value &&
      !ak_is_hermitian) {
      end_of_while = 1;
      step = &DoubleShiftQrStep<StepMatrix>;
      deflate = &DeflateSchur<Derived>;
  } else {
    step = &ImplicitQrStep<typename MatrixType::Scalar, ak_is_hermitian,
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
  return CalcEigenvaluesFromSchur<DataType>(a_matrix, ak_is_hermitian);
}

/* Calculate the eigenvalues of a Matrix using the QR decomposition
 * Parameter:
 * - a_matrix: Square Matrix
 * - ak_tol: Tolerance for considering a value 0
 * Return: Unordered Vector of (complex) eigenvalues
 */
template <typename Derived, class DataType = double>
typename std::enable_if_t<std::is_arithmetic<typename Derived::Scalar>::value,
          std::vector<std::complex<DataType>>>
QrMethod(const Eigen::MatrixBase<Derived> &ak_matrix,
         const double ak_tol = 1e-14) {
  assert(ak_matrix.rows() == ak_matrix.cols());
  static_assert(std::is_convertible<typename Derived::Scalar, DataType>::value,
      "Matrix Elements must be convertible to DataType");
  typedef Eigen::Matrix<DataType, -1, -1> Matrix;

  const bool k_is_symmetric = IsHermitian(ak_matrix, 1e-8);
  Matrix A = ak_matrix;
  Matrix p = HessenbergTransformation<>(A, ak_tol, k_is_symmetric);
  if (k_is_symmetric) {                                                           // Necessary because it is a template parameter
    return QrIterationHessenberg<std::complex<DataType>, true>(A, ak_tol);
  } else {
    return QrIterationHessenberg<std::complex<DataType>, false>(A, ak_tol);
  }
}


template <typename Derived, class DataType = std::complex<double>>
typename std::enable_if_t<IsComplex<typename Derived::Scalar>(),
  std::vector<DataType>> QrMethod(const Eigen::MatrixBase<Derived> &ak_matrix,
                                  const double ak_tol = 1e-10) {
  assert(ak_matrix.rows() == ak_matrix.cols());
  static_assert(std::is_convertible<typename Derived::Scalar, DataType>::value,
      "Matrix Elements must be convertible to DataType");
  typedef Eigen::Matrix<DataType, -1, -1> Matrix;

  const bool k_is_hermitian = IsHermitian(ak_matrix, 1e-8);
  Matrix A = ak_matrix;
  Matrix p = HessenbergTransformation<>(A, ak_tol, k_is_hermitian);

  if (k_is_hermitian) {                                                           // Necessary because it is a template parameter
    return QrIterationHessenberg<DataType, true>(A.real(), ak_tol);
  } else {
    return QrIterationHessenberg<DataType, false>(A, ak_tol);
  }
}

} // namespace nla_exam
#endif
