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
/* Determine if the given Matrix is symmetric
 * Parameter:
 * - aA: Matrix
 * Return: 'true', if aA is symmetric, 'false' else
 */
template <class Derived>
  std::enable_if_t<!is_complex<typename Derived::Scalar>(), bool>
is_symmetric(const Eigen::MatrixBase<Derived> &aA,
                  const double tol = 1e-14) {
  for (int i = 0; i < aA.rows(); ++i) {
    for (int ii = 0; ii < aA.rows(); ++ii) {
      if (std::abs(aA(i, ii) - aA(ii, i)) >= tol)
        return false;                                                             // The Matrix is not symmetric
    }
  }
  return true;
}

template <class Derived>
  std::enable_if_t<is_complex<typename Derived::Scalar>(), bool>
is_symmetric(const Eigen::MatrixBase<Derived> &aA,
                  const double tol = 1e-14) {
  for (int i = 0; i < aA.rows(); ++i) {
    for (int ii = 0; ii < aA.rows(); ++ii) {
      if (std::abs(aA(i, ii) - std::conj(aA(ii, i))) >= tol)
        return false;                                                             // The Matrix is not symmetric
    }
  }
  return true;
}

/* Create a Householder Reflection
 * Parameter:
 * - x: Vector determining the reflection
 * - P: Returns the Householder Matrix
 * Return: void
 */
template <class Derived, class Derived2> inline
std::enable_if_t<std::is_arithmetic<typename Derived::Scalar>::value, void>
create_householder(const Eigen::MatrixBase<Derived> &x,
                        const Eigen::MatrixBase<Derived2> &P) {
  typedef typename Eigen::MatrixBase<Derived2> Mat;
  typedef typename Eigen::Matrix<typename Derived::Scalar, -1, -1> Matrix;

  // TODO check for correction
  // TODO replace 1e-14 with a parameter aTol
  if( x.squaredNorm() <= 1e-14 ) return;
  Matrix u = x;
  typename Derived::Scalar alpha = - u.norm();
  if( u(0) < 0 ) alpha = -alpha;
  u(0) = u(0) - alpha;
  const_cast<Mat&>(P).setIdentity();
  const_cast<Mat&>(P)(Eigen::lastN(u.rows()), Eigen::lastN(u.rows())) -=
                      2 * (u * u.transpose()) / u.squaredNorm();                  // Calculate the relevant block
  return;
}

template <class Derived, class Derived2> inline
std::enable_if_t<is_complex<typename Derived::Scalar>(), void>
create_householder(const Eigen::MatrixBase<Derived> &x,
                        const Eigen::MatrixBase<Derived2> &P) {
  // TODO check for correctness
  typedef typename Derived::Scalar data_type;
  typedef typename Eigen::MatrixBase<Derived2> Mat;
  typedef typename Eigen::Matrix<typename Derived::Scalar, -1, -1> Matrix;
  // TODO replace 1e-14 with a parameter aTol
  if( x.squaredNorm() <= 1e-14 ) return;

  data_type alpha = std::exp(data_type{0.0,1.0} * arg(x(0)));
  Matrix u = x + alpha * x.norm() * Matrix::Identity(x.rows(), 1);
  const_cast<Mat&>(P).setIdentity();
  const_cast<Mat&>(P)(Eigen::lastN(u.rows()), Eigen::lastN(u.rows())) -=
    (2.0 / (u.adjoint() * u )(0)) * (u * u.adjoint());
  return;
}

/* Transforms a Matrix to Hessenberg form
 * Parameter:
 * - aA: Matrix to transform
 * Return: The unitary matrix used in the similarity trasnformation
 */
template <class Derived>
Eigen::Matrix<typename Derived::Scalar, -1, -1>
hessenberg_transformation(const Eigen::MatrixBase<Derived> &aA) {
  typedef Eigen::Matrix<typename Derived::Scalar, -1, -1> Matrix;
  typedef Eigen::MatrixBase<Derived> Mat;

  Matrix Q = Matrix::Identity(aA.rows(), aA.cols());                              // Q is transformation Matrix
  Matrix P(aA.rows(), aA.rows());                                                 // P is Householder reflection
  for (int i = 0; i < aA.rows() - 1; ++i) {
    create_householder(aA(Eigen::lastN(aA.rows() - i - 1), i), P);                // Calc Householder Matrix
    const_cast<Mat&>(aA) = P.adjoint() * aA * P;                                  // Transformation Step
    const_cast<Mat&>(aA)( Eigen::lastN(aA.rows() - i - 2), i) =
                      Matrix::Zero(aA.rows() - i - 2, 1);                         // Set Round off errors to 0
    Q *= P;                                                                       // Build the transformation Matrix
  }
  return Q;
}


/* Get Wilkinson shift parameter for a given 2x2 Matrix
 * Parameter:
 * - aA: 2x2 Matrix of which to calculate the shift parameter
 * Return: value of the shift parameter
 */
template <class data_type, class Derived> inline
std::enable_if_t<std::is_arithmetic<data_type>::value, data_type>
wilkinson_shift(const Eigen::MatrixBase<Derived> &aA) {
  data_type d = (aA(0, 0) - aA(1, 1)) / 2.0;
  return aA(1, 1) + d - signum(d) * std::sqrt(d * d + aA(1, 0) * aA(1, 0));
}


template <class data_type, class Derived>
std::enable_if_t<std::is_same<data_type, std::complex<typename
  Derived::Scalar::value_type>>::value, data_type>
wilkinson_shift(const Eigen::MatrixBase<Derived> &aA) {
    data_type tmp0 = aA.trace();
    data_type tmp1 = aA.determinant();
    data_type tmp = std::sqrt(tmp0 * tmp0 - 4.0 * tmp1);
    data_type ev1 = (tmp0 + tmp) / 2.0;
    data_type ev2 = (tmp0 - tmp) / 2.0;
  if (std::abs(ev1 - aA(1, 1)) < std::abs(ev2 - aA(1, 1))) {
    return ev1;
  } else {
    return ev2;
  }
}


/* Applies a single givens rotation to a tridiagonal Matrix
 * Parameter:
 *  - aA:       Triagonal Matrix
 *  - aK:       Index of upper row in the rotation
 *  - aC:       Parameter 'c' in the givens rotation
 *  - aS:       Parameter 's' in the givens rotation
 *  - aBuldge:  Current Value of the buldge
 */
template <class data_type, bool is_symmetric, class Derived>
// For complex this is not entirely correct, bc of conj or at least needs to be checked again
std::enable_if_t<is_symmetric && std::is_arithmetic<data_type>::value, void>
apply_givens(const Eigen::MatrixBase<Derived> &A, const int aK,
                              const data_type aC, const data_type aS,
                               const data_type, data_type &aBuldge) {
  Eigen::MatrixBase<Derived> &aA = const_cast<Eigen::MatrixBase<Derived> &>(A);
  data_type alpha;
  data_type beta;
  // Previous Column
  if (aK > 0) {
    alpha = aA(aK, aK - 1);
    beta = aBuldge;
    aA(aK, aK - 1) = (aC * alpha) + (aS * beta);
    aBuldge = -aS * alpha + aC * beta;
  }
  // Center Block
  alpha = aA(aK, aK);
  data_type alpha_2 = aA(aK + 1, aK + 1);
  beta = aA(aK + 1, aK);
  aA(aK, aK) = aC * aC * alpha + aS * aS * alpha_2 + 2.0 * aC * aS * beta;
  aA(aK + 1, aK) =
      -aS * (aC * alpha + aS * beta) + aC * (aC * beta + aS * alpha_2);
  aA(aK + 1, aK + 1) = aC * aC * alpha_2 + aS * aS * alpha - 2.0 * aC * aS * beta;
  // Next Column
  if (aK < aA.rows() - 2) {
    alpha = 0;                                                                    // new buldge
    beta = aA(aK + 2, aK + 1);
    aBuldge = aC * alpha + aS * beta;
    aA(aK + 2, aK + 1) = -aS * alpha + aC * beta;
  }
  return;
}

template <class data_type,  bool is_symmetric, class Derived>
std::enable_if_t<!is_symmetric, void>
apply_givens(const Eigen::MatrixBase<Derived> &aA, const int aK,
                              const data_type aC, const data_type aS,
                              const data_type aSconj) {
  typedef Eigen::Matrix<data_type, -1, -1> Matrix;
  typedef Eigen::MatrixBase<Derived> Mat;
  Matrix Q = Matrix::Identity(2, 2);
  Q(0 , 0) = aC;
  Q(1 , 1) = aC;
  Q(0 , 1) = aS;
  Q(1 , 0) = -aSconj;
  int start = std::max(0, aK -1);
  long end = std::min(long{aK + 2}, long{aA.rows() - 1});
  const_cast<Mat&>(aA)(Eigen::seq(aK, aK+1), Eigen::seq(start, aA.rows() -1)) = Q.adjoint() * aA(Eigen::seq(aK, aK+1), Eigen::seq(start, aA.rows() -1));
  const_cast<Mat&>(aA)(Eigen::seq(0, end), Eigen::seq(aK, aK+1)) = aA(Eigen::seq(0, end), Eigen::seq(aK, aK+1)) * Q;
  return;
}

template <class data_type, bool is_symmetric, class Derived> inline
// For complex this is not entirely correct, bc of conj or at least needs to be checked again
std::enable_if_t<is_symmetric && !std::is_arithmetic<data_type>::value, void>
apply_givens(const Eigen::MatrixBase<Derived> &A, const int aK,
                              const data_type aC, const data_type aS,
                              const data_type aSconj, data_type &aBuldge) {
  apply_givens<data_type, false>(A, aK, aC, aS, aSconj);
  //aBuldge = A(aK+1, aK-1);
  return;
}



/* Calculate the entries of a Givens Matrix
 * Parameter:
 * - a: first entry
 * - b: entry to eliminate
 * Return: Vector containing {c, s, std::conj(s)}
 */
template<class data_type>
std::enable_if_t<std::is_arithmetic<data_type>::value, std::vector<data_type>>
get_givens_entries(const data_type& a, const data_type& b) {
  std::vector<data_type> res(3);
  data_type r = std::hypot(a, b);
  res.at(0) = a / r;
  res.at(1) = b / r;
  res.at(2) = res.at(1);
  return res;
}


template<class data_type> inline
std::enable_if_t<is_complex<data_type>(), std::vector<data_type>>
get_givens_entries(const data_type& a, const data_type& b) {
  assert( 1 != 0);
  typedef typename data_type::value_type real;
  std::vector<data_type> res(3);
  real absa = std::abs(a);
  real absb = std::abs(b);
  real r = std::hypot(absa, absb);
  res.at(0) = absa / r;
  res.at(1) = -a / absa * (std::conj(b) / r);
  res.at(2) = std::conj(res.at(1));
  return res;
}


/* Executes one step of the implicit qr algorithm for a tridiagonal Matrix
 * Parameter:
 * - aA: Tridiagonal Matrix
 * Return: void
 */
template <class data_type, bool is_symmetric, typename Derived>
std::enable_if_t<is_symmetric, void>
implicit_qr_step(const Eigen::MatrixBase<Derived> &aA) {
  data_type shift = wilkinson_shift<data_type>(aA(Eigen::lastN(2), Eigen::lastN(2)));
  data_type buldge = 0;
  auto entries = get_givens_entries(aA(0, 0) - shift, aA(1, 0));
  apply_givens<data_type, is_symmetric>(aA, 0, entries.at(0), entries.at(1),      // Initial step
      entries.at(2), buldge);

  for (int k = 1; k < aA.rows() - 1; ++k) {                                       // Buldge Chasing
  //entries = get_givens_entries(aA(k, k-1), buldge);
  entries = get_givens_entries(aA(k, k-1), aA(k+1, k-1));
  apply_givens<data_type, is_symmetric>(aA, k, entries.at(0), entries.at(1),      // Initial step
      entries.at(2), buldge);
//    if (std::abs(buldge) < 1e-14)
//      break;
  }
  return;
}


template <class data_type, bool is_symmetric, typename Derived>
std::enable_if_t<!is_symmetric, void>
implicit_qr_step(const Eigen::MatrixBase<Derived> &aA) {
  data_type shift = wilkinson_shift<data_type>(aA(Eigen::lastN(2),
        Eigen::lastN(2)));
  auto entries = get_givens_entries(aA(0, 0) - shift, aA(1, 0));
  apply_givens<data_type, false>(aA, 0, entries.at(0), entries.at(1),             // Initial step
      entries.at(2));

  for (int k = 1; k < aA.rows() - 1; ++k) {                                       // Buldge Chasing
  entries = get_givens_entries(aA(k, k-1), aA(k+1, k-1));
  apply_givens<data_type, false>(aA, k, entries.at(0), entries.at(1),             // Initial step
      entries.at(2));
    //if (std::abs(buldge) < 1e-14)
//    if (std::abs(aA(k+1, k-1)) < 1e-14)
//      break;
  }
  return;
}



/* Get double shift parameters for a given 2x2 Matrix
 * Parameter:
 * - aA: 2x2 Matrix of which to calculate the shift parameter
 * Return: vector containing both double shift parameters
 */
template <class Derived>
typename std::enable_if_t<std::is_arithmetic<typename Derived::Scalar>::value,
std::vector<typename Derived::Scalar>>
double_shift_parameter(const Eigen::MatrixBase<Derived> &aA) {
  typedef typename Derived::Scalar data_type;
  std::vector<typename Eigen::MatrixBase<Derived>::Scalar> res(2);
  //  If Real use the same but with the eigenvalues
  res.at(0) = -aA.trace();
  res.at(1) = aA.determinant();
  // TODO implicit shift when possible?
  if( res.at(0) * res.at(0) > 4.0 * res.at(1) ) {
    data_type tmp = std::sqrt(res.at(0) * res.at(0) - 4.0 * res.at(1));
    data_type ev1 = (-res.at(0) + tmp) / 2.0;
    data_type ev2 = (-res.at(0) - tmp) / 2.0;
    if (std::abs(ev1 - aA(1,1)) < std::abs(ev2 - aA(1,1))) {
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
typename std::enable_if_t<is_complex<typename Derived::Scalar>(),
std::vector<typename Derived::Scalar>>
double_shift_parameter(const Eigen::MatrixBase<Derived> &aA) {
  typedef typename Derived::Scalar data_type;
  data_type ev = wilkinson_shift<data_type>(aA(Eigen::lastN(2), Eigen::lastN(2)));
  std::vector<data_type> res(2);
  res.at(0) = -2.0 * ev;
  res.at(1) = ev * ev;
  return res;
}


// TODO Optimize using the known Matrix structure
/* Executes one step of the double shift algorithm
 * Parameter:
 * - aA: Tridiagonal Matrix
 * Return: void
 */
template <class Derived>
void double_shift_qr_step(const Eigen::MatrixBase<Derived> &aA) {
  typedef Eigen::MatrixBase<Derived> Mat;
  typedef Eigen::Matrix<typename Derived::Scalar, -1, -1> Matrix;
  int n = aA.rows();
  std::vector<typename Derived::Scalar> shift =
      double_shift_parameter(aA(Eigen::lastN(2), Eigen::lastN(2)));
  Matrix m1 = aA * aA(Eigen::all, 0) + shift.at(0) * aA(Eigen::all, 0)
              + shift.at(1) * Matrix::Identity(n, 1);                             // Only compute the first col
  Matrix P(n,n);                                                                  // Householder Matrix

  create_householder(m1, P);                                                      // Calc initial Step
  const_cast<Eigen::MatrixBase<Derived> &>(aA) = P.adjoint() * aA * P;            // Transformation Step

  for (int i = 0; i < n - 1; ++i) {
    create_householder(aA(Eigen::seq(i + 1, n - 1), i), P);                       // Buldge Chasing
    const_cast<Mat&>(aA) = P.adjoint() * aA * P;                                  // Transformation Step
    const_cast<Mat&>(aA)( Eigen::seq(i + 2, n - 1), i) =
                    Matrix::Zero(n - i - 2, 1);                                   // Set Round off errors to 0
  }
}


// TODO Eventually return threads
/* Deflates a Matrix converging to a diagonal matrix
 * Parameter:
 * - aA: Matrix to deflate
 * - aBegin: Index of fhe block that is solved currently
 * - aEnd: Index of fhe End that is solved currently
 * - aTol: Tolerance for considering a value 0
 * Return: "true" if the block is fully solved, "false" otherwise
 */
template <class Derived>
bool deflate_diagonal(const Eigen::MatrixBase<Derived> &aA, int &aBegin,
                      int &aEnd, const double aTol = 1e-14) {
  bool state = true;
  for (int i = aEnd; i > aBegin; --i) {
    if (std::abs(aA(i, i - 1)) < aTol * std::abs(aA(i, i) + aA(i - 1, i - 1))) {
      const_cast<Eigen::MatrixBase<Derived> &>(aA)(i, i - 1) = 0;
      if (!state) {
        aBegin = i;
        return false;                                                             // Subblock to solve found
      }
    } else if (state) {                                                           // Start of the block found
      aEnd = i;
      state = false;
    }
  }
  return state;
}


/* Deflates a Matrix converging to a Schur Matrix
 * Parameter:
 * - aA: Matrix to deflate
 * - aBegin: Index of fhe block that is solved currently
 * - aEnd: Index of fhe End that is solved currently
 * - aTol: Tolerance for considering a value 0
 * Return: "true" if the block is fully solved, "false" otherwise
 */
template <class Derived>
bool deflate_schur(const Eigen::MatrixBase<Derived> &aA, int &aBegin,
                   int &aEnd, const double aTol = 1e-14) {
  bool state = true;
  for (int i = aEnd; i > aBegin; --i) {
    if (std::abs(aA(i, i - 1)) < aTol * std::abs(aA(i, i) + aA(i - 1, i - 1))) {
      const_cast<Eigen::MatrixBase<Derived> &>(aA)(i, i - 1) = 0;
      if (!state) {
        aBegin = i;
        return false;                                                             // Subblock to solve found
      }
    } else if (state && (i - 1 > aBegin) &&
               (std::abs(aA(i - 1, i - 2)) >=
                aTol * std::abs(aA(i - 2, i - 2) + aA(i - 1, i - 1)))) {          // Start of the block found
      aEnd = i;
      --i;                                                                        // Next index already checked
      state = false;
    }
  }
  return state;
}


// TODO Needs changes for complex EVs
/* Calculates the eigenvalues of a Matrix in Schur Form
 * Parameter:
 * - aA: Matrix in Schur Form
 * - aMatrix_is_diagonal: "true" if aA is diagonal, "false" otherwise
 * Return: Unordered Vector of eigenvalues
 */
template <class data_type, class Derived>
std::vector<data_type>
calc_eigenvalues_from_schur(const Eigen::MatrixBase<Derived>& aA,
                            bool aMatrix_is_diagonal = false) {
  std::vector<data_type> res(aA.rows());
  if (aMatrix_is_diagonal) {
    for (int i = 0; i < aA.rows(); ++i) {
      res.at(i) = aA(i, i);
    }
  } else {
    for (int i = 0; i < aA.rows() - 1; ++i) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-equal"
      if (std::abs(aA(i + 1, i)) == 0) {                                        // Eigenvalue in diagonal block
#pragma GCC diagnostic pop
        res.at(i) = aA(i, i);
      } else {                                                                  // Eigenvalue in a 2x2 block
        typename Derived::Scalar d = (aA(i, i) + aA(i + 1, i + 1)) / 2.0;
        data_type pq = std::sqrt(data_type{ d * d - aA(i, i) * aA(i + 1, i + 1)
              + aA(i, i + 1) * aA(i + 1, i)});
        res.at(i) = d - pq;                                                     // First eigenvalue
        ++i;
        res.at(i) = d + pq;                                                     // Second eigenvalue
      }
    }
#pragma GCC diagnostic ignored "-Wfloat-equal"
#pragma GCC diagnostic push
    if (std::abs(aA(aA.rows() - 1, aA.rows() - 2)) == 0.0)                      // Last EV is in a diagonal block
#pragma GCC diagnostic pop
      res.at(aA.rows() - 1) = aA(aA.rows() - 1, aA.rows() - 1);                 // Add the last eigenvalue
  }
  return res;
}


/* Get the eigenvalues of a hessenberg Matrix using the qr iteration
 * Parameter:
 * - aA: Hessenberg Matrix
 * - aIs_symmetric: "true" if aA is symmetric, "false" otherwise
 * - aTol: Tolerance for considering a value 0
 * Return: Unordered Vector of eigenvalues
 */
template <class data_type, class Derived>
std::vector<data_type>
qr_iteration_hessenberg(const Eigen::MatrixBase<Derived> &aA,
                        const bool aIs_symmetric = false,
                        const double aTol = 1e-14) {
  typedef Eigen::MatrixBase<Derived> Mat;
  typedef Eigen::Block<Derived, -1, -1, false> step_Matrix;
  int begin = 0;
  int end = aA.rows() - 1;
  int end_of_while;
  std::vector<std::complex<double>> res;
  void (*step)(const Eigen::MatrixBase<step_Matrix> &);
  bool (*deflate)(const Eigen::MatrixBase<Derived> &, int &, int &,
                  const double);

  if (aIs_symmetric || is_complex<typename Derived::Scalar>()) {
    end_of_while = 0;
    if (aIs_symmetric) {
      step = &implicit_qr_step<typename Mat::Scalar, true, step_Matrix>;
    } else {
      step = &implicit_qr_step<typename Mat::Scalar, false, step_Matrix>;
      }
    deflate = &deflate_diagonal<Derived>;
  } else {
    end_of_while = 1;
    step = &double_shift_qr_step<step_Matrix>;
    deflate = &deflate_schur<Derived>;
  }

  while (end_of_while < end) {
    if (deflate(aA, begin, end, aTol)) {
      end = begin - 1;
      begin = 0;
    } else {
      step(const_cast<Mat&>(aA)(Eigen::seq(begin, end),
              Eigen::seq(begin, end)));
    }
  }
  return calc_eigenvalues_from_schur<data_type>(aA, aIs_symmetric);
}

/* Calculate the eigenvalues of a Matrix using the QR decomposition
 * Parameter:
 * - aA: Square Matrix
 * - aTol: Tolerance for considering a value 0
 * Return: Unordered Vector of (complex) eigenvalues
 */
template <typename Derived, class data_type = double>
typename std::enable_if_t<std::is_arithmetic_v<typename Derived::Scalar>,
          std::vector<std::complex<data_type>>>
qr_method(const Eigen::MatrixBase<Derived> &aA, const double aTol = 1e-14) {
  assert(aA.rows() == aA.cols());
  const bool a_is_symmetric = is_symmetric(aA);
  typedef Eigen::Matrix<data_type, -1, -1> Matrix;
  Matrix A = aA;
  Matrix P = hessenberg_transformation(A);
  return qr_iteration_hessenberg<std::complex<data_type>>(A, a_is_symmetric, aTol);
}


template <typename Derived, class data_type = std::complex<double>>
typename std::enable_if_t<is_complex<typename Derived::Scalar>(), std::vector<data_type>>
qr_method(const Eigen::MatrixBase<Derived> &aA, const double aTol = 1e-10) {
  assert(aA.rows() == aA.cols());
  const bool a_is_symmetric = is_symmetric(aA);
  typedef Eigen::Matrix<data_type, -1, -1> Matrix;
  Matrix A = aA;
  Matrix P = hessenberg_transformation(A);

  Eigen::MatrixXd M_real(A.rows(), A.cols());
  for(int i = 0; i < A.rows(); ++i) {
    for(int j = 0; j < A.cols(); ++j) {
      M_real(i, j) = std::abs(A(i,j));
    }
  }

  return qr_iteration_hessenberg<data_type>(A, a_is_symmetric, aTol);
}

} // namespace nla_exam
#endif
