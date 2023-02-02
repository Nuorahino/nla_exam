#ifndef QR_HH_
#define QR_HH_
/*
 * QR Method for
 *https://www.cs.cornell.edu/~bindel/class/cs6210-f12/notes/lec28.pdf
 *
 */

/*
 * TODO: optimize for eigen (noalias)
 * TODO: Assert that data_type and Matrix are compatible
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
std::enable_if_t<std::is_arithmetic<typename Derived::Scalar>::value, void>
CreateHouseholder(const Eigen::MatrixBase<Derived> &ak_x,
                  const Eigen::MatrixBase<Derived2> &a_p,
                  const double ak_tol = 1e-14) {
  typedef typename Eigen::MatrixBase<Derived2> MatrixType2;
  typedef typename Eigen::Matrix<typename Derived::Scalar, -1, -1> MatrixType1;

  // TODO check for correction
  // TODO in function calls pass tolerance
  //if( ak_x.squaredNorm() <= ak_tol ) return;
  if( ak_x.squaredNorm() <= ak_tol * ak_tol ) return;
  MatrixType1 u = ak_x;
  typename Derived::Scalar alpha = - u.norm();
  if( u(0) < 0 ) alpha = -alpha;
  u(0) = u(0) - alpha;
  const_cast<MatrixType2&>(a_p).setIdentity();
  const_cast<MatrixType2&>(a_p)(Eigen::lastN(u.rows()), Eigen::lastN(u.rows())) -=
                      2 * (u * u.transpose()) / u.squaredNorm();                  // Calculate the relevant block
  return;
}

template <class Derived, class Derived2> inline
std::enable_if_t<IsComplex<typename Derived::Scalar>(), void>
CreateHouseholder(const Eigen::MatrixBase<Derived> &ak_x,
                  const Eigen::MatrixBase<Derived2> &a_p,
                  const double ak_tol = 1e-14) {
  // TODO check for correctness
  typedef typename Derived::Scalar data_type;
  typedef typename Eigen::Matrix<data_type, -1, -1> MatrixType1;
  typedef typename Eigen::MatrixBase<Derived2> MatrixType2;
  // TODO replace 1e-14 with a parameter ak_tol
  if( ak_x.squaredNorm() <= ak_tol ) return;

  data_type alpha = std::exp(data_type{0.0,1.0} * arg(ak_x(0)));
  MatrixType1 u = ak_x + alpha * ak_x.norm() *
    MatrixType1::Identity(ak_x.rows(), 1);
  const_cast<MatrixType2&>(a_p).setIdentity();
  const_cast<MatrixType2&>(a_p)(Eigen::lastN(u.rows()),
      Eigen::lastN(u.rows())) -= (2.0 / (u.adjoint() * u )(0)) *
    (u * u.adjoint());
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
                         const double ak_tol = 1e-14) {
  typedef Eigen::Matrix<typename Derived::Scalar, -1, -1> Matrix;
  typedef Eigen::MatrixBase<Derived> MatrixType;

  Matrix q = Matrix::Identity(a_matrix.rows(), a_matrix.cols());                  // q is transformation Matrix
  Matrix p(a_matrix.rows(), a_matrix.rows());                                     // p is Householder reflection
  for (int i = 0; i < a_matrix.rows() - 1; ++i) {
    CreateHouseholder(a_matrix(Eigen::lastN(a_matrix.rows() - i - 1), i), p,
        ak_tol);                                                                  // Calc Householder Matrix
    const_cast<MatrixType&>(a_matrix) = p.adjoint() * a_matrix * p;                      // Transformation Step
    const_cast<MatrixType&>(a_matrix)( Eigen::lastN(a_matrix.rows() - i - 2), i) =
                      Matrix::Zero(a_matrix.rows() - i - 2, 1);                   // Set Round off errors to 0
//    const_cast<MatrixType&>(a_matrix)(i, Eigen::lastN(a_matrix.rows() - i - 2)) =
//                      Matrix::Zero(a_matrix.rows() - i - 2, 1);                   // Set Round off errors to 0
    q *= p;                                                                       // Build the transformation Matrix
  }
  return q;
}


/* Get Wilkinson shift parameter for a given 2x2 Matrix
 * Parameter:
 * - ak_matrix: 2x2 Matrix of which to calculate the shift parameter
 * Return: value of the shift parameter
 */
template <class data_type, class Derived> inline
std::enable_if_t<std::is_arithmetic<data_type>::value, data_type>
WilkinsonShift(const Eigen::MatrixBase<Derived> &ak_matrix) {
  data_type d = (ak_matrix(0, 0) - ak_matrix(1, 1)) / 2.0;
  if( d >= 0 ) {
    return ak_matrix(1, 1) + d - std::sqrt(d * d + ak_matrix(1, 0) *
        ak_matrix(1, 0));
  } else {
    return ak_matrix(1, 1) + d + std::sqrt(d * d + ak_matrix(1, 0) *
        ak_matrix(1, 0));
  }
}

// TODO is complex number check correct?

template <class data_type, class Derived>
std::enable_if_t<IsComplex<data_type>(), data_type>
WilkinsonShift(const Eigen::MatrixBase<Derived> &ak_matrix) {
    data_type trace = ak_matrix.trace();
    data_type tmp = std::sqrt(trace * trace - 4.0 * ak_matrix.determinant());
    data_type ev1 = (trace + tmp) / 2.0;
    data_type ev2 = (trace - tmp) / 2.0;
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
template <class data_type, bool is_symmetric, class Derived>
// For complex this is not entirely correct,
// bc of conj or at least needs to be checked again
std::enable_if_t<is_symmetric && std::is_arithmetic<data_type>::value, void>
ApplyGivens(const Eigen::MatrixBase<Derived> &a_matrix, const int ak_k,
             const data_type ak_c, const data_type ak_s, const data_type,
             data_type &a_buldge) {
  Eigen::MatrixBase<Derived> &matrix = const_cast<Eigen::MatrixBase<Derived> &>(a_matrix);
  data_type alpha;
  data_type beta;
  // Previous Column
  if (ak_k > 0) {
    alpha = matrix(ak_k, ak_k - 1);
    beta = a_buldge;
    matrix(ak_k, ak_k - 1) = (ak_c * alpha) + (ak_s * beta);
    a_buldge = -ak_s * alpha + ak_c * beta;
  }
  // Center Block
  alpha = matrix(ak_k, ak_k);
  data_type alpha_2 = matrix(ak_k + 1, ak_k + 1);
  beta = matrix(ak_k + 1, ak_k);
  matrix(ak_k, ak_k) = ak_c * ak_c * alpha + ak_s * ak_s * alpha_2 + 2.0 * ak_c * ak_s * beta;
  matrix(ak_k + 1, ak_k) =
      -ak_s * (ak_c * alpha + ak_s * beta) + ak_c * (ak_c * beta + ak_s * alpha_2);
  matrix(ak_k + 1, ak_k + 1) = ak_c * ak_c * alpha_2 + ak_s * ak_s * alpha - 2.0 * ak_c * ak_s * beta;
  // Next Column
  if (ak_k < matrix.rows() - 2) {
    alpha = 0;                                                                    // new buldge
    beta = matrix(ak_k + 2, ak_k + 1);
    a_buldge = ak_c * alpha + ak_s * beta;
    matrix(ak_k + 2, ak_k + 1) = -ak_s * alpha + ak_c * beta;
  }
  return;
}

template <class data_type,  bool is_symmetric, class Derived>
std::enable_if_t<!is_symmetric, void>
ApplyGivens(const Eigen::MatrixBase<Derived> &a_matrix, const int ak_k,
            const data_type ak_c, const data_type ak_s,
            const data_type ak_sconj) {
  typedef Eigen::Matrix<data_type, -1, -1> Matrix;
  typedef Eigen::MatrixBase<Derived> MatrixType;
  Matrix Q = Matrix::Identity(2, 2);
  Q(0 , 0) = ak_c;
  Q(1 , 1) = ak_c;
  Q(0 , 1) = ak_s;
  Q(1 , 0) = -ak_sconj;
  int start = std::max(0, ak_k -1);
  long end = std::min(long{ak_k + 2}, long{a_matrix.rows() - 1});
  const_cast<MatrixType&>(a_matrix)(Eigen::seq(ak_k, ak_k+1), Eigen::seq(start,
        a_matrix.rows() -1)) = Q.adjoint() * a_matrix(Eigen::seq(ak_k, ak_k+1),
      Eigen::seq(start, a_matrix.rows() -1));
  const_cast<MatrixType&>(a_matrix)(Eigen::seq(0, end), Eigen::seq(ak_k, ak_k+1)) =
    a_matrix(Eigen::seq(0, end), Eigen::seq(ak_k, ak_k+1)) * Q;
  return;
}

template <class data_type, bool is_symmetric, class Derived> inline
// For complex this is not entirely correct, bc of conj or at least needs to be checked again
std::enable_if_t<is_symmetric && !std::is_arithmetic<data_type>::value, void>
ApplyGivens(const Eigen::MatrixBase<Derived> &A, const int ak_k,
                              const data_type ak_c, const data_type ak_s,
                              const data_type ak_sconj, data_type &a_buldge) {
  ApplyGivens<data_type, false>(A, ak_k, ak_c, ak_s, ak_sconj);
  //a_buldge = A(ak_k+1, ak_k-1);
  return;
}



/* Calculate the entries of a Givens Matrix
 * Parameter:
 * - ak_a: first entry
 * - ak_b: entry to eliminate
 * Return: Vector containing {c, s, std::conj(s)}
 */
template<class data_type>
std::enable_if_t<std::is_arithmetic<data_type>::value, std::vector<data_type>>
GetGivensEntries(const data_type& ak_a, const data_type& ak_b) {
  std::vector<data_type> res(3);
  data_type r = std::hypot(ak_a, ak_b);
  res.at(0) = ak_a / r;
  res.at(1) = ak_b / r;
  res.at(2) = res.at(1);
  return res;
}

template<class data_type> inline
std::enable_if_t<IsComplex<data_type>(), std::vector<data_type>>
GetGivensEntries(const data_type& ak_a, const data_type& ak_b) {
  assert( 1 != 0);
  typedef typename data_type::value_type real;
  std::vector<data_type> res(3);
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
template <class data_type, bool is_symmetric, typename Derived>
std::enable_if_t<is_symmetric, void>
ImplicitQrStep(const Eigen::MatrixBase<Derived> &a_matrix) {
  data_type shift = WilkinsonShift<data_type>(a_matrix(Eigen::lastN(2),
        Eigen::lastN(2)));
  data_type buldge = 0;
  auto entries = GetGivensEntries(a_matrix(0, 0) - shift, a_matrix(1, 0));
  ApplyGivens<data_type, is_symmetric>(a_matrix, 0, entries.at(0),
      entries.at(1), entries.at(2), buldge);                                            // Initial step

  for (int k = 1; k < a_matrix.rows() - 1; ++k) {                                       // Buldge Chasing
  // TODO find one working for both
    entries = GetGivensEntries(a_matrix(k, k-1), buldge);
  //entries = GetGivensEntries(a_matrix(k, k-1), a_matrix(k+1, k-1));
  ApplyGivens<data_type, is_symmetric>(a_matrix, k, entries.at(0),
      entries.at(1), entries.at(2), buldge);
  //TODO  Maybe neccesary
//    if (std::abs(buldge) < 1e-14)
//      break;
  }
  return;
}

template <class data_type, bool is_symmetric, typename Derived>
std::enable_if_t<!is_symmetric, void>
ImplicitQrStep(const Eigen::MatrixBase<Derived> &a_matrix) {
  data_type shift = WilkinsonShift<data_type>(a_matrix(Eigen::lastN(2),
        Eigen::lastN(2)));
  auto entries = GetGivensEntries(a_matrix(0, 0) - shift, a_matrix(1, 0));
  ApplyGivens<data_type, false>(a_matrix, 0, entries.at(0), entries.at(1),             // Initial step
      entries.at(2));

  for (int k = 1; k < a_matrix.rows() - 1; ++k) {                                       // Buldge Chasing
  entries = GetGivensEntries(a_matrix(k, k-1), a_matrix(k+1, k-1));
  ApplyGivens<data_type, false>(a_matrix, k, entries.at(0), entries.at(1),
      entries.at(2));
  //TODO  Maybe neccesary
    //if (std::abs(buldge) < 1e-14)
//    if (std::abs(a_matrix(k+1, k-1)) < 1e-14)
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
  typedef typename Derived::Scalar data_type;
  std::vector<typename Eigen::MatrixBase<Derived>::Scalar> res(2);
  //  If Real use the same but with the eigenvalues
  res.at(0) = -ak_matrix.trace();
  res.at(1) = ak_matrix.determinant();
  // TODO implicit shift when possible?
  if( res.at(0) * res.at(0) > 4.0 * res.at(1) ) {
    data_type tmp = std::sqrt(res.at(0) * res.at(0) - 4.0 * res.at(1));
    data_type ev1 = (-res.at(0) + tmp) / 2.0;
    data_type ev2 = (-res.at(0) - tmp) / 2.0;
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

template <class Derived> inline
typename std::enable_if_t<IsComplex<typename Derived::Scalar>(),
std::vector<typename Derived::Scalar>>
DoubleShiftParameter(const Eigen::MatrixBase<Derived> &ak_matrix) {
  typedef typename Derived::Scalar data_type;
  data_type ev = WilkinsonShift<data_type>(ak_matrix(Eigen::lastN(2),
        Eigen::lastN(2)));
  std::vector<data_type> res(2);
  res.at(0) = -2.0 * ev;
  res.at(1) = ev * ev;
  return res;
}


// TODO Optimize using the known Matrix structure
/* Executes one step of the double shift algorithm
 * Parameter:
 * - a_matrix: Tridiagonal Matrix
 * Return: void
 */
template <class Derived>
void DoubleShiftQrStep(const Eigen::MatrixBase<Derived> &a_matrix) {
  typedef Eigen::MatrixBase<Derived> MatrixType;
  typedef Eigen::Matrix<typename Derived::Scalar, -1, -1> Matrix;
  int n = a_matrix.rows();
  std::vector<typename Derived::Scalar> shift =
      DoubleShiftParameter(a_matrix(Eigen::lastN(2), Eigen::lastN(2)));
  Matrix m1 = a_matrix * a_matrix(Eigen::all, 0) + shift.at(0) *
    a_matrix(Eigen::all, 0) + shift.at(1) * Matrix::Identity(n, 1);               // Only compute the first col
  Matrix p(n,n);                                                                  // Householder Matrix
  CreateHouseholder(m1, p);                                                       // Calc initial Step
  const_cast<MatrixType &>(a_matrix) = p.adjoint() * a_matrix * p;                // Transformation Step
  //for (int i = 0; i < n - 1; ++i) {
  for (int i = 0; i < n - 2; ++i) {
    CreateHouseholder(a_matrix(Eigen::seq(i + 1, n - 1), i), p);                  // Buldge Chasing
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
    if (std::abs(a_matrix(i, i - 1)) < ak_tol * std::abs(a_matrix(i, i) + a_matrix(i - 1, i - 1))) {
      const_cast<Eigen::MatrixBase<Derived> &>(a_matrix)(i, i - 1) = 0;
      if (!state) {
        a_begin = i;
        return false;                                                             // Subblock to solve found
      }
    } else if (state && (i - 1 > a_begin) &&
               (std::abs(a_matrix(i - 1, i - 2)) >=
                ak_tol * std::abs(a_matrix(i - 2, i - 2) + a_matrix(i - 1, i - 1)))) {          // Start of the block found
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
template <class data_type, class Derived>
std::vector<data_type>
CalcEigenvaluesFromSchur(const Eigen::MatrixBase<Derived>& ak_matrix,
                         const bool ak_matrix_is_diagonal = false) {
  std::vector<data_type> res(ak_matrix.rows());
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
        data_type pq = std::sqrt(data_type{ d * d - ak_matrix(i, i) *
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
template <class data_type, class Derived>
std::vector<data_type>
QrIterationHessenberg(const Eigen::MatrixBase<Derived> &a_matrix,
                        const bool ak_is_hermitian = false,
                        const double ak_tol = 1e-14) {
  typedef Eigen::MatrixBase<Derived> MatrixType;
  typedef Eigen::Block<Derived, -1, -1, false> step_Matrix;
  int begin = 0;
  int end = a_matrix.rows() - 1;
  int end_of_while;
  std::vector<std::complex<double>> res;
  void (*step)(const Eigen::MatrixBase<step_Matrix> &);
  bool (*deflate)(const Eigen::MatrixBase<Derived> &, int &, int &,
                  const double);

  if (ak_is_hermitian || IsComplex<typename Derived::Scalar>()) {
    end_of_while = 0;
    if (ak_is_hermitian) {
      step = &ImplicitQrStep<typename MatrixType::Scalar, true, step_Matrix>;
    } else {
      step = &ImplicitQrStep<typename MatrixType::Scalar, false, step_Matrix>;
      }
    deflate = &DeflateDiagonal<Derived>;
  } else {
    end_of_while = 1;
    step = &DoubleShiftQrStep<step_Matrix>;
    deflate = &DeflateSchur<Derived>;
  }

  while (end_of_while < end) {
    if (deflate(a_matrix, begin, end, ak_tol)) {
      end = begin - 1;
      begin = 0;
    } else {
      step(const_cast<MatrixType&>(a_matrix)(Eigen::seq(begin, end),
              Eigen::seq(begin, end)));
    }
  }
  return CalcEigenvaluesFromSchur<data_type>(a_matrix, ak_is_hermitian);
}

/* Calculate the eigenvalues of a Matrix using the QR decomposition
 * Parameter:
 * - a_matrix: Square Matrix
 * - ak_tol: Tolerance for considering a value 0
 * Return: Unordered Vector of (complex) eigenvalues
 */
template <typename Derived, class data_type = double>
typename std::enable_if_t<std::is_arithmetic<typename Derived::Scalar>::value,
          std::vector<std::complex<data_type>>>
QrMethod(const Eigen::MatrixBase<Derived> &a_matrix, const double ak_tol = 1e-14) {
  assert(a_matrix.rows() == a_matrix.cols());
  const bool a_is_symmetric = IsHermitian(a_matrix, 1e-8);
  typedef Eigen::Matrix<data_type, -1, -1> Matrix;
  Matrix A = a_matrix;
  Matrix p = HessenbergTransformation(A, ak_tol);
  return QrIterationHessenberg<std::complex<data_type>>(A, a_is_symmetric, ak_tol);
}


template <typename Derived, class data_type = std::complex<double>>
typename std::enable_if_t<IsComplex<typename Derived::Scalar>(), std::vector<data_type>>
QrMethod(const Eigen::MatrixBase<Derived> &a_matrix, const double ak_tol = 1e-10) {
  assert(a_matrix.rows() == a_matrix.cols());
  const bool a_is_symmetric = IsHermitian(a_matrix, 1e-8);
  typedef Eigen::Matrix<data_type, -1, -1> Matrix;
  Matrix A = a_matrix;
  Matrix p = HessenbergTransformation(A, ak_tol);

//  Eigen::MatrixXd M_real(A.rows(), A.cols());
//  for(int i = 0; i < A.rows(); ++i) {
//    for(int j = 0; j < A.cols(); ++j) {
//      M_real(i, j) = std::abs(A(i,j));
//    }
//  }

  return QrIterationHessenberg<data_type>(A, a_is_symmetric, ak_tol);
}

} // namespace nla_exam
#endif
