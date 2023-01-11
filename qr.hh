#ifndef QR_HH
#define QR_HH
/*
 * QR Method for
 *https://www.cs.cornell.edu/~bindel/class/cs6210-f12/notes/lec28.pdf
 *
 */

/*
 * TODO: Assert rows() == cols()
 * TODO: How should data_type be passed?
 * TODO: implement for complex
 * TODO: optimize for eigen (noalias)
 * TODO: Assert that data_type and Matrix are compatible
 */

// Notes: 1.4.12
// Transform A to Hessenberg

#include <cmath>
#include <complex>
#include <type_traits>
#include <vector>
#include <iostream>

#include <eigen3/Eigen/Dense>

#include "helpfunctions/helpfunctions.hh"

/* Determine if the given Matrix is symmetric
 * Parameter:
 * - aA: Matrix
 * Return: 'true', if aA is symmetric, 'false' else
 */
template <class Derived>
bool is_symmetric(const Eigen::MatrixBase<Derived> &aA,
                  const double tol = 1e-14) {
  for (int i = 0; i < aA.rows(); ++i) {
    for (int ii = 0; ii < aA.rows(); ++ii) {
      if (std::abs(aA(i, ii) - aA(ii, i)) >= tol)
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
template <class Derived, class Derived2>
void create_householder(const Eigen::MatrixBase<Derived> &x,
                        const Eigen::MatrixBase<Derived2> &P) {
  typedef typename Eigen::MatrixBase<Derived2> Mat;
  typedef typename Eigen::Matrix<typename Derived::Scalar, -1, -1> Matrix;

  Matrix u = x + signum(x(0, 0)) * x.norm() * Matrix::Identity(x.rows(), 1);
  const_cast<Mat&>(P).setIdentity();
  const_cast<Mat&>(P)(Eigen::lastN(u.rows()), Eigen::lastN(u.rows())) -=
                      2 * (u * u.transpose()) / u.squaredNorm();                  // Calculate the relevant block
  return;
}


/* Transforms a Matrix to Hessenberg form
 * Parameter:
 * - aA: Matrix to transform
 * Return: The unitary matrix used in the similarity trasnformation
 */
template <class Derived>
Eigen::Matrix<typename Derived::Scalar, -1, -1>
hessenberg_transformation(const Eigen::MatrixBase<Derived> &aA,
                          const bool aIs_symmetric) {
  typedef Eigen::Matrix<typename Derived::Scalar, -1, -1> Matrix;
  typedef Eigen::MatrixBase<Derived> Mat;

  Matrix Q = Matrix::Identity(aA.rows(), aA.cols());                              // Q is transformation Matrix
  Matrix P(aA.rows(), aA.rows());                                                 // P is Householder reflection
  for (int i = 0; i < aA.rows() - 1; ++i) {
    create_householder(aA(Eigen::lastN(aA.rows() - i - 1), i), P);                // Calc Householder Matrix
    const_cast<Mat&>(aA) = P.transpose() * aA * P;                                // Transformation Step
    const_cast<Mat&>(aA)( Eigen::lastN(aA.rows() - i - 2), i) =
                      Matrix::Zero(aA.rows() - i - 2, 1);                         // Set Round off errors to 0
    Q *= P;                                                                       // Build the transformation Matrix
  }
  return Q;
}


/* Get Wilkinsion shift parameter for a given 2x2 Matrix
 * Parameter:
 * - aA: 2x2 Matrix of which to calculate the shift parameter
 * Return: value of the shift parameter
 */
template <class data_type, class Derived>
data_type wilkinson_shift(const Eigen::MatrixBase<Derived> &aA) {
  data_type d = (aA(0, 0) - aA(1, 1)) / 2;
  return aA(1, 1) + d - signum(d) * std::sqrt(d * d + aA(1, 0) * aA(1, 0));
}


/* Applies a single givens rotation to a tridiagonal Matrix
 * Parameter:
 *  - aA:       Triagonal Matrix
 *  - aK:       Index of upper row in the rotation
 *  - aC:       Parameter 'c' in the givens rotation
 *  - aS:       Parameter 's' in the givens rotation
 *  - aBuldge:  Current Value of the buldge
 */
template <class data_type, class Derived>
void apply_givens_tridiagonal(const Eigen::MatrixBase<Derived> &A, const int aK,
                              const data_type aC, const data_type aS,
                              data_type &aBuldge) {
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
  aA(aK, aK) = aC * aC * alpha + aS * aS * alpha_2 + 2 * aC * aS * beta;
  aA(aK + 1, aK) =
      -aS * (aC * alpha + aS * beta) + aC * (aC * beta + aS * alpha_2);
  aA(aK + 1, aK + 1) = aC * aC * alpha_2 + aS * aS * alpha - 2 * aC * aS * beta;
  // Next Column
  if (aK < aA.rows() - 2) {
    alpha = 0;                                                                    // new buldge
    beta = aA(aK + 2, aK + 1);
    aBuldge = aC * alpha + aS * beta;
    aA(aK + 2, aK + 1) = -aS * alpha + aC * beta;
  }
  return;
}


/* Executes one step of the implicit qr algorithm for a tridiagonal Matrix
 * Parameter:
 * - aA: Tridiagonal Matrix
 * Return: void
 */
template <class data_type, typename Derived>
void implicit_qr_step_tridiagonal(const Eigen::MatrixBase<Derived> &aA) {
  data_type shift = wilkinson_shift<data_type>(aA(Eigen::lastN(2), Eigen::lastN(2)));
  // TODO Maybe use aA * aA instead
  data_type r =
      std::sqrt(std::pow(aA(0, 0) - shift, 2) + std::pow(aA(1, 0), 2));
  data_type c = (aA(0, 0) - shift) / r;
  data_type s = aA(1, 0) / r;
  data_type buldge = 0;

  apply_givens_tridiagonal<data_type>(aA, 0, c, s, buldge);                       // Initial step
  for (int k = 1; k < aA.rows() - 1; ++k) {                                       // Buldge Chasing
    r = std::sqrt(std::pow(aA(k, k - 1), 2) + std::pow(buldge, 2));
    c = (aA(k, k - 1)) / r;
    s = buldge / r;
    apply_givens_tridiagonal<data_type>(aA, k, c, s, buldge);
//    if (std::abs(buldge) < 1e-14)
//      break;
  }
  return;
}


// TODO implicit shift when possible?
/* Get double shift parameters for a given 2x2 Matrix
 * Parameter:
 * - aA: 2x2 Matrix of which to calculate the shift parameter
 * Return: vector containing both double shift parameters
 */
template <class Derived>
std::vector<typename Eigen::MatrixBase<Derived>::Scalar>
double_shift_parameter(const Eigen::MatrixBase<Derived> &aA) {
  std::vector<typename Eigen::MatrixBase<Derived>::Scalar> res(2);
  //  If Real use the same but with the eigenvalues
  res.at(0) = -aA.trace();
  res.at(1) = aA.determinant();
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
  Matrix M = aA * aA + shift.at(0) * aA + shift.at(1) * Matrix::Identity(n, n);
  Matrix P(n,n);                                                                  // Householder Matrix

  create_householder(M(Eigen::all, 0), P);                                        // Calc initial Step
  const_cast<Eigen::MatrixBase<Derived> &>(aA) = P.transpose() * aA * P;          // Transformation Step

  for (int i = 0; i < n - 1; ++i) {
    create_householder(aA(Eigen::seq(i + 1, n - 1), i), P);                       // Buldge Chasing
    const_cast<Mat&>(aA) = P.transpose() * aA * P;                                // Transformation Step
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
    if (std::abs(aA(i, i - 1)) < aTol * std::abs(aA(i, i) + aA(i - 1, i - 1))) {  //
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
      if (aA(i + 1, i) == 0) {                                                  // Eigenvalue in diagonal block
        res.at(i) = aA(i, i);
      } else {                                                                  // Eigenvalue in a 2x2 block
        typename Derived::Scalar d = (aA(i, i) + aA(i + 1, i + 1)) / 2;
        data_type pq = std::sqrt(data_type( d * d - aA(i, i) * aA(i + 1, i + 1)
              + aA(i, i + 1) * aA(i + 1, i)));
        res.at(i) = d - pq;                                                     // First eigenvalue
        ++i;
        res.at(i) = d + pq;                                                     // Second eigenvalue
      }
    }
    if (aA(aA.rows() - 1, aA.rows() - 2) == 0)                                  // Last EV is in a diagonal block
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
  void (*step_func)(const Eigen::MatrixBase<step_Matrix> &);
  bool (*deflate)(const Eigen::MatrixBase<Derived> &, int &, int &,
                  const double);

  if (aIs_symmetric) {
    end_of_while = 0;
    step_func = &implicit_qr_step_tridiagonal<typename Mat::Scalar, step_Matrix>;
    deflate = &deflate_diagonal<Derived>;
  } else {
    end_of_while = 1;
    step_func = &double_shift_qr_step<step_Matrix>;
    deflate = &deflate_schur<Derived>;
  }

  while (end_of_while < end) {
    if (deflate(aA, begin, end, aTol)) {
      end = begin - 1;
      begin = 0;
    } else {
      step_func(const_cast<Mat&>(aA)(Eigen::seq(begin, end), Eigen::seq(begin, end)));
    }
  }
  if (aIs_symmetric) {
    res = calc_eigenvalues_from_schur<data_type>(aA, true);
  } else {
    res = calc_eigenvalues_from_schur<data_type>(aA);
  }
  return res;
}

// TODO Write the rest better
//// Default imtplementation
template <
    typename Derived,
    class data_type = typename std::enable_if<
        !std::is_arithmetic<typename Derived::Scalar>::value, double>::type>
std::vector<std::complex<data_type>>
qr_method(const Eigen::EigenBase<Derived> &aA) {
  Eigen::VectorXd res;
  std::cout << typeid(typename Derived::Scalar).name() << std::endl;
  std::cout << std::is_arithmetic<typename Derived::Scalar>::value << std::endl;
  std::cout << "Incompatible Matrix: Matrix must have arithmatic type"
            << std::endl; // Replace with cerr
}

template <typename Derived, class data_type = double> // First argument is Matrix type, second is type of the
                      // calculations
typename std::enable_if< std::is_arithmetic<typename Derived::Scalar>::value, std::vector<std::complex<data_type>>>::type
implicit_shift_qr_method( const Eigen::MatrixBase<Derived> &aA) {
  typedef Eigen::Matrix<data_type, Eigen::Dynamic, Eigen::Dynamic> Matrix; // Maybe change this
  Matrix A = aA;
  Matrix P = hessenberg_transformation(A, true);
  std::vector<std::complex<data_type>> res = qr_iteration_hessenberg<std::complex<data_type>>(A, true);
  return res;
}

template <typename Derived,
          class data_type =
              double> // First argument is Matrix type, second is type of the
                      // calculations
                      typename std::enable_if<
                          std::is_arithmetic<typename Derived::Scalar>::value,
                          std::vector<std::complex<data_type>>>::type
                      double_shift_qr_method(
                          const Eigen::MatrixBase<Derived> &aA) {
  typedef Eigen::Matrix<data_type, Eigen::Dynamic, Eigen::Dynamic>
      Matrix; // Maybe change this
  Matrix A = aA;
  Matrix P = hessenberg_transformation(A, false);
  std::vector<std::complex<data_type>> res = qr_iteration_hessenberg<std::complex<data_type>>(A, false);
  return res;
}

// TODO Check for template, not correct for complex eigenvalues
// TODO remove the enable if from the double
template <typename Derived,
          class data_type = typename std::enable_if<
              std::is_arithmetic<typename Derived::Scalar>::value, double>::
              type> // First argument is Matrix type, second is type of the
                    // calculations
                    // typename std::enable_if<std::is_arithmetic<typename
                    // Derived::Scalar>::value, Eigen::Matrix<data_type,
                    // Eigen::Dynamic, 1>>::type
                    std::vector<std::complex<data_type>>
                    // auto
                    qr_method(const Eigen::MatrixBase<Derived> &aA) {
  const bool a_is_symmetric =
      is_symmetric(aA); // Needs to be changed, or in the upper one
  if (a_is_symmetric)
    return implicit_shift_qr_method<Derived, data_type>(aA);
  else
    return double_shift_qr_method<Derived, data_type>(aA);
}

#endif
