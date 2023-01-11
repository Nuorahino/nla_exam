#ifndef QR_HH
#define QR_HH
/*
 * QR Method for
 *https://www.cs.cornell.edu/~bindel/class/cs6210-f12/notes/lec28.pdf
 *
 * Required Implementations in the Matrix class:
 *                                              - Matrix(row, col)
 *                                              - Matrix.rows()
 *                                              - Matrix.transpose()
 *                                              - Matrix::Identity(rows, cols)
 */

/*
 * TODO: Assert rows() == cols()
 */

// Notes: 1.4.12
// Transform A to Hessenberg

#include <vector>
#include <cmath>
#include <type_traits>
#include <complex>

#include <eigen3/Eigen/Dense>                   // Definition of 'seq' and 'lastN'
#include <eigen3/Eigen/Sparse>                   // Temporary
#include <iostream>

#include "helpfunctions/helpfunctions.hh"

/* Determine if the given Matrix is symmetric
 * Parameter:
 * - aA: Matrix
 * Return: 'true', if aA is symmetric, 'false' else
 */
template<class Derived>
bool is_symmetric(const Eigen::MatrixBase<Derived>& aA, const double tol = 1e-14) {
  for( int i = 0; i < aA.rows(); ++i ) {
    for( int ii = 0; ii < aA.rows(); ++ii) {
      if( std::abs( aA(i, ii) - aA(ii, i)) >= tol ) return false;                       // The Matrix is not symmetric
    }
  }
  return true;
}


template<class Derived, class Matrix>
void create_householder(const Eigen::MatrixBase<Derived>& x, const int aN, Matrix& P) {
  Matrix u = x + signum(x(0,0)) * x.norm()
    * Matrix::Identity(x.rows(),1);
  P = Matrix::Identity(aN,aN);                                                      // Increase Matrix with Identity
  P(Eigen::lastN(u.rows()), Eigen::lastN(u.rows())) -=                              // Calculate the relevant block
      2 * (u * u.transpose()) / u.squaredNorm();
  return;
}


// Current implementation requires Eigen Dense
// Currently requires: Matrix.norm(), Indexing by Matrix(Eigen::lastN(n), i)
template<class Matrix>
Matrix hessenberg_transformation(Matrix& aA, const bool aIs_symmetric) {
  Matrix Q = Matrix::Identity(aA.rows(), aA.cols());                                    // Q is transformation Matrix
  Matrix P;                                                                             // P is Householder reflection
  for( int i = 0; i < aA.rows()-1; ++i) {
    create_householder(aA(Eigen::lastN(aA.rows()-i-1), i), aA.rows(), P);               // Calc Householder Matrix
    aA = P.transpose() * aA * P;                                                        // Transformation Step
    aA(Eigen::lastN(aA.rows()-i-2), i) = Matrix::Zero(aA.rows() - i - 2, 1);            // Set Round off errors to 0
    if( aIs_symmetric ) {
      aA(i, Eigen::lastN(aA.rows()-i-2)) = Matrix::Zero(1, aA.rows() - i - 2);          // Restore symmetry
    }
    Q = Q * P;                                                                          // Build the transformation Matrix
  }
  return Q;
}


// Symmetric QR Iteration Step

// Real Symmetric Matrix

/* Get Wilkinsion shift parameter for a given Matrix
 * Parameter:
 * - aA: Matrix of which to calculate the shift parameter
 * - n: last row, of the current block (default aA.rows()-1)
 * Return: value of the shift parameter
 */
template<class Derived, class data_type = typename Derived::Scalar> inline
data_type wilkinson_shift(const Eigen::MatrixBase<Derived>& aA, const int n) {
  data_type d = (aA(n-1, n-1) - aA(n,n)) / 2;
  return aA(n,n) + d - signum(d) * std::sqrt(d*d + aA(n,n-1) * aA(n, n-1));
}

template<class Derived, class data_type = typename Derived::Scalar> inline
data_type wilkinson_shift(const Eigen::MatrixBase<Derived>& aA) {
  return wilkinson_shift(aA, aA.rows()-1);
}


/* Applying a single givens rotation to a tridiagonal Matrix
 * Parameter:
 *  - aA:       Triagonal Matrix
 *  - aK:       Index of upper row in the rotation
 *  - aC:       Parameter 'c' in the givens rotation
 *  - aS:       Parameter 's' in the givens rotation
 *  - aBuldge:  Current Value of the buldge
 */
template<class Derived, class data_type>
void apply_givens_tridiagonal(const Eigen::MatrixBase<Derived>& A, const int aK, const data_type aC, const data_type aS, data_type& aBuldge) {
  Eigen::MatrixBase<Derived>& aA = const_cast<Eigen::MatrixBase<Derived>&>(A);
  data_type alpha;
  data_type beta;
  //Previous Column
  if( aK > 0 ) {
    alpha = aA(aK, aK-1);
    beta = aBuldge;
    aA(aK, aK-1) = (aC * alpha) + (aS * beta);
    aBuldge = - aS * alpha + aC * beta;
  }
  // Center Block
  alpha = aA(aK, aK);
  data_type alpha_2 = aA(aK+1, aK+1);
  beta = aA(aK+1, aK);
  aA(aK, aK) = aC * aC * alpha + aS * aS * alpha_2  + 2 * aC * aS * beta;
  aA(aK+1, aK) = -aS * (aC * alpha + aS * beta) + aC * (aC * beta + aS * alpha_2);
  aA(aK+1, aK+1) = aC * aC * alpha_2 + aS * aS * alpha - 2 * aC * aS * beta;
  // Next Colume
  if( aK < aA.rows() - 2 ) {
    alpha = 0;                                                                      // new buldge
    beta = aA(aK+2, aK+1);
    aBuldge = aC * alpha + aS * beta;
    aA(aK+2, aK+1) = -aS * alpha + aC * beta;
  }
  return;
}

template <typename Derived, class data_type = typename Derived::Scalar>
void givens_step_tridiagonal(const Eigen::MatrixBase<Derived>& aA) {
  data_type shift = wilkinson_shift(aA);
  data_type r = std::sqrt(std::pow(aA(0,0)-shift,2) + std::pow(aA(1, 0),2));
  data_type c = (aA(0, 0)-shift)/r;
  data_type s = aA(1, 0)/r;
  data_type buldge = 0;

  apply_givens_tridiagonal(aA, 0, c, s, buldge);                                  // Initial step
  for (int k = 1; k < aA.rows()-1; ++k) {                                         // Buldge Chasing
    r = std::sqrt(std::pow(aA(k, k-1),2) + std::pow(buldge,2));
    c = (aA(k, k-1))/r;
    s = buldge/r;
    apply_givens_tridiagonal(aA, k, c, s, buldge);
    if (std::abs(buldge) < 1e-14) break;
  }
  return;  // Probably needs a return
}

//
// Non Symmetric QR Iteration Step
// TODO implicit shift when possible?
template<class Derived>
std::vector<typename Eigen::MatrixBase<Derived>::Scalar> double_shift_parameter(const Eigen::MatrixBase<Derived>& aA) {
  std::vector<typename Eigen::MatrixBase<Derived>::Scalar> res(2);
  //  If Real use the same but with the eigenvalues
  res.at(0) = -aA.trace();
  res.at(1) = aA.determinant();
  return res;
}


template<class Derived>
void householder_transformation(const Eigen::MatrixBase<Derived>& aA) {
  typedef Eigen::MatrixXd Matrix;
  int n = aA.rows();
  Matrix P;                                                                         // Householder Matrix
  for( int i = 0; i < n-1; ++i) {
    create_householder(aA(Eigen::seq(i+1, n - 1), i), n, P);                        // Calc Householder Matrix
  const_cast<Eigen::MatrixBase<Derived>&>(aA) = P.transpose() * aA * P;             // Transformation Step
  const_cast<Eigen::MatrixBase<Derived>&>(aA)(Eigen::seq(i+2, aA.rows()-1), i) = Matrix::Zero(aA.rows() - i - 2, 1);       // Set Round off errors to 0
  }
}


// Some changes take effect outside of the block
template<class Derived>
void explicit_double_shift_qr_step(const Eigen::MatrixBase<Derived>& aA) {
  typedef Eigen::Matrix<typename Derived::Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
  int n = aA.rows();
  std::vector<typename Matrix::Scalar> shift = double_shift_parameter(aA(Eigen::lastN(2), Eigen::lastN(2)));
  Matrix M = aA * aA + shift.at(0) * aA + shift.at(1) * Matrix::Identity(n,n);
  std::cout << "M" << M << std::endl;
  Matrix P;                                                                             // Householder Matrix
  create_householder(M(Eigen::all, 0), n, P);                  // Calc Householder Matrix
  const_cast<Eigen::MatrixBase<Derived>&>(aA) = P.transpose() * aA * P;                 // Transformation Step

  householder_transformation(aA);
}


// Eventually return threads // Needs to be changed for double shift
template<class Derived>
bool deflate_simple(const Eigen::MatrixBase<Derived>& aA, int& aBegin, int& aEnd, const double aTol = 1e-14) {
  bool state = true;
  for( int i = aEnd; i > aBegin; --i ) {
    if( std::abs(aA(i, i-1)) < aTol * std::abs(aA(i, i) + aA(i-1, i-1)) ) {
      const_cast<Eigen::MatrixBase<Derived>&>(aA)(i, i-1) = 0;
      if( !state ) {
        aBegin = i;
        return false ;
      }
    } else if( state ) {
      aEnd = i;
      state = false;
    }
  }
  return state;
}


template<class Derived>
bool deflate_double(const Eigen::MatrixBase<Derived>& aA, int& aBegin, int& aEnd, const double aTol = 1e-14) {
  bool state = true;
  for( int i = aEnd; i > aBegin; --i ) {
    //std::cout << i << std::endl;
    //std::cout <<  state << (i - 1 > aBegin) << ((i-1 > aBegin) && (std::abs(aA(i-1, i-2)) < aTol * std::abs(aA(i-2, i-2) + aA(i-1, i-1))))  << std::endl;
    if( std::abs(aA(i, i-1)) < aTol * std::abs(aA(i, i) + aA(i-1, i-1)) ) {
      std::cout << "Setting to 0" << std::endl;
      const_cast<Eigen::MatrixBase<Derived>&>(aA)(i, i-1) = 0;
      if( !state ) {
        aBegin = i;
        return false ;
      }
    } else if( state && (i - 1 > aBegin) && (std::abs(aA(i-1, i-2)) >= aTol * std::abs(aA(i-2, i-2) + aA(i-1, i-1))) ) {
      std::cout << " element found" << std::endl;
      aEnd = i;
      --i;
      state = false;
    }
  }
  return state;
}


template<class Derived, class data_type = typename Derived::Scalar>
std::vector<std::complex<data_type>>
calc_eigenvalues_from_schur(Eigen::MatrixBase<Derived>& aA, bool real_ev = false) {
  std::vector<std::complex<data_type>> res(aA.rows());
  if( real_ev ) {
    for( int i = 0; i < aA.rows(); ++i ) {
      res.at(i) = aA(i,i);
    }
  } else {
    for( int i = 0; i < aA.rows()-1; ++i ) {
      if( aA(i+1, i) == 0 ) {
        res.at(i) = aA(i,i);
      }
      else {
        // Calc 2x2 eigenvalues maybe correct
        data_type d = (aA(i, i) + aA(i+1, i+1)) / 2;
        std::complex<data_type> pq = std::sqrt(std::complex<data_type>(d*d - aA(i,i) * aA(i+1, i+1) + aA(i,i+1) * aA(i+1, i)));
        res.at(i) = d + pq;
        ++i;
        res.at(i) = d - pq;
      }
    }
    if( aA(aA.rows()-1,aA.rows()-2) == 0 ) res.at(aA.rows()-1) = aA(aA.rows()-1, aA.rows()-1); // Add the last eigenvalue
  }
  return res;
}


// Assumes dimensions to match and classes to be compatible
// A is Hessenberg
template <class Derived, class data_type = typename Derived::Scalar> // Unclear yet how it should be passed
std::vector<std::complex<data_type>>
hessenberg_qr_iteration(Eigen::MatrixBase<Derived>& aA, const int aBegin, int aEnd, const bool aIs_symmetric, const double aTol = 1e-14) {
  typedef typename Eigen::Block<Derived, -1, -1, false> step_Matrix;
  int begin = aBegin;
  int end_of_while;
  std::vector<std::complex<double>> res;
  void (*step_func)(const Eigen::MatrixBase<step_Matrix>&);
  bool (*deflate)(const Eigen::MatrixBase<Derived>&, int&, int&, const double);
    if( aIs_symmetric ) {
      end_of_while = aBegin;
      step_func = &givens_step_tridiagonal<step_Matrix>;
      deflate = &deflate_simple<Derived>;
    } else {
      end_of_while = aBegin +1;
      step_func = &explicit_double_shift_qr_step<step_Matrix>;
      deflate = &deflate_double<Derived>;
  }

  std::cout << aA << std::endl;
  while( end_of_while < aEnd ) {
    std::cout << "Before Step" << std::endl;
    std::cout << aA << std::endl;
    if( deflate(aA, begin, aEnd, aTol)) {
    //if(false) {
      aEnd = begin - 1;
      begin = aBegin;
    } else {
      std::cout << "step" << std::endl;
      std::cout << "current Block" << std::endl;
      std::cout << aA(Eigen::seq(begin, aEnd), Eigen::seq(begin, aEnd)) << std::endl;
      step_func(aA(Eigen::seq(begin,aEnd), Eigen::seq(begin, aEnd)));
    }
  }
  if( aIs_symmetric ) {
    res = calc_eigenvalues_from_schur(aA, true);
  } else {
    res = calc_eigenvalues_from_schur(aA);
  }
  return res;
}

// Function w/o aBegin and aEnd
template <class Derived, class data_type = typename Derived::Scalar>
std::vector<std::complex<data_type>>
hessenberg_qr_iteration(Eigen::MatrixBase<Derived>& aA, const bool aIs_symmetric, const double aTol = 1e-14) {
  return hessenberg_qr_iteration(aA, 0, aA.rows()-1, aIs_symmetric, aTol);
}

//// Default imtplementation
template <typename Derived, class data_type = typename std::enable_if<!std::is_arithmetic<typename Derived::Scalar>::value, double>::type>
std::vector<std::complex<data_type>>
qr_method(const Eigen::EigenBase<Derived>& aA) {
  Eigen::VectorXd res;
  std::cout << typeid(typename Derived::Scalar).name() << std::endl;
  std::cout << std::is_arithmetic<typename Derived::Scalar>::value << std::endl;
  std::cout << "Incompatible Matrix: Matrix must have arithmatic type" << std::endl; // Replace with cerr
}


template <typename Derived, class data_type = double> // First argument is Matrix type, second is type of the calculations
typename std::enable_if<std::is_arithmetic<typename Derived::Scalar>::value, std::vector<std::complex<data_type>>>::type
implicit_shift_qr_method(const Eigen::MatrixBase<Derived>& aA) {
  typedef Eigen::Matrix<data_type, Eigen::Dynamic, Eigen::Dynamic> Matrix; // Maybe change this
  Matrix A = aA;
  Matrix A2 = aA; // TODO remove
  Matrix P = hessenberg_transformation(A, true);
  std::vector<std::complex<data_type>> res = hessenberg_qr_iteration(A, true);
  std::cout << "Non Symm" << std::endl;
  hessenberg_transformation(A2, false);       // TODO remove
  res = hessenberg_qr_iteration(A2, false); // TODO remove
  return res;
}


template <typename Derived, class data_type = double> // First argument is Matrix type, second is type of the calculations
typename std::enable_if<std::is_arithmetic<typename Derived::Scalar>::value, std::vector<std::complex<data_type>>>::type
double_shift_qr_method(const Eigen::MatrixBase<Derived>& aA) {
  std::vector<std::complex<data_type>> res(aA.rows());
  typedef Eigen::Matrix<data_type, Eigen::Dynamic, Eigen::Dynamic> Matrix; // Maybe change this
  Matrix A = aA;
  Matrix P = hessenberg_transformation(A, false);
  res = hessenberg_qr_iteration(A, false);
  return res;
}

// TODO Check for template, not correct for complex eigenvalues
// TODO remove the enable if from the double
template <typename Derived, class data_type = typename std::enable_if<std::is_arithmetic<typename Derived::Scalar>::value, double>::type > // First argument is Matrix type, second is type of the calculations
//typename std::enable_if<std::is_arithmetic<typename Derived::Scalar>::value, Eigen::Matrix<data_type, Eigen::Dynamic, 1>>::type
std::vector<std::complex<data_type>>
//auto
qr_method(const Eigen::MatrixBase<Derived>& aA) {
  std::cout << "correct Method" << std::endl;
  const bool a_is_symmetric = is_symmetric(aA);  // Needs to be changed, or in the upper one
  if( a_is_symmetric ) return implicit_shift_qr_method<Derived, data_type>(aA);
  else return double_shift_qr_method<Derived, data_type>(aA);
}


#endif
