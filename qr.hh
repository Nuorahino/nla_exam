#ifndef QR_HH
#define QR_HH
/*
 * QR Method for
 *
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
template<class Matrix>
bool is_symmetric(const Matrix& aA, const double tol = 1e-14) {
  for( int i = 0; i < aA.rows(); ++i ) {
    for( int ii = 0; ii < aA.rows(); ++ii) {
      if( std::abs( aA(i, ii) - aA(ii, i)) >= tol ) return false;                       // The Matrix is not symmetric
    }
  }
  return true;
}


// Currently requires: Matrix.squaredNorm(), Indexing by Matrix(Eigen::lastN(n), Eigen::lastN(n))
template<class Matrix> inline
Matrix create_householder(const Matrix& aU, const int aN, const bool aIs_symmetric) {
  Matrix res = Matrix::Identity(aN,aN);                                                 // Increase Matrix with Identity
  res(Eigen::lastN(aU.rows()), Eigen::lastN(aU.rows())) -=                              // Calculate the relevant block
      2 * (aU * aU.transpose()) / aU.squaredNorm();
  return res;
}


// Current implementation requires Eigen Dense
// Currently requires: Matrix.norm(), Indexing by Matrix(Eigen::lastN(n), i)
template<class Matrix>
Matrix hessenberg_transformation(Matrix& aA, const bool aIs_symmetric) {
  Matrix Q = Matrix::Identity(aA.rows(), aA.cols());                                    // Q is transformation Matrix
  for( int i = 0; i < aA.rows()-1; ++i) {
    Matrix x = aA(Eigen::lastN(aA.rows()-i-1), i);                                      // Tmp Var for readible code
    Matrix u = x + signum(aA(i+1,i)) * x.norm()
      * Matrix::Identity(aA.rows()-i-1,1);
    Matrix P = create_householder(u, aA.rows(), aIs_symmetric);                         // Calc Householder Matrix
    aA = P * aA * P;                                                                    // Transformation Step
    aA(Eigen::lastN(aA.rows()-i-2), i) = Matrix::Zero(aA.rows() - i - 2, 1);            // Set Round off errors to 0
    if( aIs_symmetric ) {
      aA(i, Eigen::lastN(aA.rows()-i-2)) = Matrix::Zero(1, aA.rows() - i - 2);          // Keep symmetry
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
template<class Matrix, class data_type> inline
data_type wilkinson_shift(const Matrix& aA, const int n) {
  data_type d = (aA(n-1, n-1) - aA(n,n)) / data_type(2);
  return aA(n,n) + d - data_type(signum(d)) * std::sqrt(d*d + aA(n,n-1) * aA(n, n-1));
}

template<class Matrix, class data_type> inline
data_type wilkinson_shift(const Matrix& aA) {
  return wilkinson_shift<Matrix>(aA, aA.rows()-1);
}


/* Applying a single givens rotation to a tridiagonal Matrix
 * Parameter:
 *  - aA:       Triagonal Matrix
 *  - aBegin:   Index of the first row in the block
 *  - aEnd:     Index of the last row in the block
 *  - aK:       Index of upper row in the rotation
 *  - aC:       Parameter 'c' in the givens rotation
 *  - aS:       Parameter 's' in the givens rotation
 *  - aBuldge:  Current Value of the buldge
 */
template<class Matrix, class data_type>
void apply_givens_tridiagonal(Matrix& aA, const int aBegin, const int aEnd, const int aK, const data_type aC, const data_type aS, data_type& aBuldge) {
  data_type alpha;
  data_type beta;
  if( aK > aBegin ) {                                                    // Adjust the old buldge, if it is not the first row
    alpha = aA(aK, aK-1);
    beta = aBuldge;
    aA(aK, aK-1) = (aC * alpha) + (aS * beta);
    aBuldge = - aS * alpha + aC * beta;
  }

  alpha = aA(aK, aK);
  data_type alpha_2 = aA(aK+1, aK+1);
  beta = aA(aK+1, aK);
  aA(aK, aK) = aC * aC * alpha + aS * aS * alpha_2  + data_type(2) * aC * aS * beta;
  aA(aK+1, aK) = -aS * (aC * alpha + aS * beta) + aC * (aC * beta + aS * alpha_2);
  aA(aK+1, aK+1) = aC * aC * alpha_2 + aS * aS * alpha - data_type(2) * aC * aS * beta;

  if( aK < aEnd - 1 ) {
    alpha = 0; // new buldge
    beta = aA(aK+2, aK+1);
    aBuldge = aC * alpha + aS * beta;
    aA(aK+2, aK+1) = -aS * alpha + aC * beta;
  }
  return;
}

template <typename Matrix, class data_type>
void givens_step_tridiagonal(Matrix& aA, const int aBegin, const int aEnd) {
  //Eigen::MatrixBase<Derived>& aA = const_cast<Eigen::MatrixBase<Derived>&>(A);
  data_type shift = wilkinson_shift<Matrix, data_type>(aA, aEnd);
  data_type r = std::sqrt(std::pow(aA(aBegin,aBegin)-shift,2) + std::pow(aA(aBegin+1, aBegin),2));
  data_type c = (aA(aBegin, aBegin)-shift)/r;
  data_type s = aA(aBegin+1, aBegin)/r;
  data_type buldge = 0;
  std::cout << "c = " << c << std::endl;
  apply_givens_tridiagonal<Matrix, data_type>(aA, 0, aA.rows()-1, 0, c, s, buldge); // TODO needs change
  for (int k = aBegin+1; k < aEnd; ++k) {
    r = std::sqrt(std::pow(aA(k, k-1),2) + std::pow(buldge,2));
    c = (aA(k, k-1))/r;
    s = buldge/r;
    apply_givens_tridiagonal<Matrix, data_type>(aA, 0, aA.rows()-1, k, c, s, buldge); // TODO needs change
    if (std::abs(buldge) < 1e-14) break;
  }
  return;  // Probably needs a return
}




// Non Symmetric QR Iteration Step
template<class Matrix, class data_type> inline
std::vector<data_type> double_shift_parameter(const Matrix& aA, const int n) {
  std::vector<data_type> res(2);
  data_type d = (aA(n-1, n-1) + aA(n,n)) / data_type(2);
  res.at(0) = d + std::sqrt(d*d + aA(n,n-1) * aA(n, n-1));
  res.at(1) = d - std::sqrt(d*d + aA(n,n-1) * aA(n, n-1));
  return res;
}

// Givens rotation for Hessenberg Matrix
template<class Matrix, class data_type>
void apply_givens_rotation(Matrix& aA, const int aBegin, const int n, const int j, const int k) {
  data_type r = std::sqrt(std::pow(aA(j, j),2) + std::pow(aA(k, j),2));
  data_type c = aA(j, j)/r;
  data_type s = aA(k,j)/r;
  Matrix Q = Matrix::Identity(n, n); // This is not the best way to implement this
  Q(j - aBegin, j - aBegin) = c;
  Q(k - aBegin, k - aBegin) = c;
  Q(j - aBegin, k - aBegin) = s;
  Q(k - aBegin, j - aBegin) = -s; // complex this is not correct
  //Q(k - aBegin, j - aBegin) = - std::conj(s); // check if correct
  aA(Eigen::seqN(aBegin, n), Eigen::seqN(aBegin, n)) = Q * aA(Eigen::seqN(aBegin, n), Eigen::seqN(aBegin, n)) * Q.transpose();
//  for(int i = aBegin; i < aBegin + n; ++i ) {
//    for(int ii = aBegin; ii < i; ++ii ) {
//      aA(i, ii) = 0;
//    }
//  }
}


template<class Matrix, class data_type>
void givens_transformation(Matrix& aA, const int aBegin, const int aEnd) {
  int n = aEnd - aBegin + 1;
  data_type r;
  data_type c;
  data_type s;
  for(int i = aBegin; i < aEnd; ++i) {
    apply_givens_rotation<Matrix, data_type>(aA, aBegin, n, i, i+1);
  std::cout << "aA: " << aA << std::endl;
  }
}


template<class Matrix, class data_type>
void explicit_double_shift_qr_step(Matrix& aA, const int aBegin, const int aEnd) {
  int n = aEnd - aBegin + 1;
  std::vector<data_type> shift = double_shift_parameter<Matrix, data_type>(aA, aEnd);
  aA(Eigen::seq(aBegin, aEnd), Eigen::seq(aBegin, aEnd)) -= shift.at(0) * Matrix::Identity(n, n);   // Shift
  givens_transformation<Matrix, data_type>(aA, aBegin, aEnd);
  aA(Eigen::seq(aBegin, aEnd), Eigen::seq(aBegin, aEnd)) += (shift.at(0) - shift.at(1)) * Matrix::Identity(n, n);
  givens_transformation<Matrix, data_type>(aA, aBegin, aEnd);
  aA(Eigen::seq(aBegin, aEnd), Eigen::seq(aBegin, aEnd)) += shift.at(1) * Matrix::Identity(n, n);
}


// Eventually return threads
template<class Matrix>
bool deflate(Matrix& aA, int& aBegin, int& aEnd, const double aTol = 1e-14) {
  bool state = true;
  for( int i = aEnd; i > aBegin; --i ) {
    if( std::abs(aA(i, i-1)) < aTol * std::abs(aA(i, i) + aA(i-1, i-1)) ) {
      aA(i, i-1) = 0;
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

// Assumes dimensions to match and classes to be compatible
// A is Hessenberg
template <class Matrix> // Unclear yet how it should be passed
void hessenberg_qr_iteration(Matrix& aA, const int aBegin, int aEnd, const bool aIs_symmetric, const double aTol = 1e-14) {
  int begin = aBegin;
  void (*step_func)(Matrix&, const int, const int);
    if( aIs_symmetric ) {
    step_func = &givens_step_tridiagonal<Matrix, typename Matrix::Scalar>;
  } else {
    step_func = &explicit_double_shift_qr_step<Matrix, typename Matrix::Scalar>;
  }

    std::cout << aA << std::endl;
    while( aBegin < aEnd ) {
      step_func(aA, begin, aEnd);
      if( deflate<Matrix>(aA, begin, aEnd, aTol)) {
        aEnd = begin;
        begin = aBegin;
        deflate<Matrix>(aA, begin, aEnd, aTol);
      }
      std::cout << "After Step" << std::endl;
      std::cout << aA << std::endl;
      std::cout << "current Block" << std::endl;
      std::cout << aA(Eigen::seq(begin, aEnd), Eigen::seq(begin, aEnd)) << std::endl;
  }
}

// Function w/o aBegin and aEnd
template <class Matrix>
void hessenberg_qr_iteration(Matrix& aA, const bool aIs_symmetric, const double aTol = 1e-14) {
  hessenberg_qr_iteration<Matrix>(aA, 0, aA.rows()-1, aIs_symmetric, aTol);
}

//// Default imtplementation
//template<class Matrix, class data_type = std::enable_if<!std::is_arithmetic<typename Matrix::Scalar>::value, double>>
///template<class Matrix, class data_type>
template <typename Derived, class data_type = double>
typename std::enable_if<!std::is_arithmetic<typename Derived::Scalar>::value, void>::type
qr_method(const Eigen::EigenBase<Derived>& aA) {
  Eigen::VectorXd res;
  std::cout << typeid(typename Derived::Scalar).name() << std::endl;
  std::cout << std::is_arithmetic<typename Derived::Scalar>::value << std::endl;
  std::cout << "Incompatible Matrix: Matrix must have arithmatic type" << std::endl; // Replace with cerr

  typedef Eigen::SparseMatrix<double> SMat;
  typename Eigen::MatrixXi::Scalar i = 2.3;

  std::cout << std::is_base_of<Eigen::EigenBase<Eigen::MatrixXd>, Eigen::MatrixXd>::value << std::endl;
  std::cout << std::is_same<Eigen::EigenBase<Eigen::MatrixXd>, Eigen::MatrixXd>::value << std::endl;
  std::cout << std::is_base_of<Eigen::SparseMatrixBase<SMat>, SMat>::value << std::endl;
  std::cout << std::is_base_of<Eigen::EigenBase<SMat>, SMat>::value << std::endl;
  std::cout << "last test" << std::endl;
  std::cout << std::is_base_of<Eigen::MatrixBase<SMat>, SMat>::value << std::endl;
  std::cout << typeid(typename Eigen::MatrixXd::Scalar).name() << std::endl;
}


template <typename Derived, class data_type = double> // First argument is Matrix type, second is type of the calculations
typename std::enable_if<std::is_arithmetic<typename Derived::Scalar>::value, Eigen::Matrix<data_type, Eigen::Dynamic, 1>>::type
implicit_shift_qr_method(const Eigen::MatrixBase<Derived>& aA) {
  typedef Eigen::Matrix<data_type, Eigen::Dynamic, Eigen::Dynamic> Matrix;
  Matrix A = aA;
  Matrix A2 = aA; // TODO remove
  Matrix P = hessenberg_transformation(A, true);
  hessenberg_qr_iteration<Matrix>(A, true);
  std::cout << "Non Symm" << std::endl;
  hessenberg_transformation(A2, false);
  hessenberg_qr_iteration<Matrix>(A2, false); // TODO remove
  Eigen::Matrix<data_type, Eigen::Dynamic, 1> res = A.diagonal();
  return res;
}


template <typename Derived, class data_type> // First argument is Matrix type, second is type of the calculations
typename std::enable_if<std::is_arithmetic<typename Derived::Scalar>::value, Eigen::Matrix<data_type, Eigen::Dynamic, 1>>::type
double_shift_qr_method(const Eigen::MatrixBase<Derived>& aA) {
  typedef Eigen::Matrix<std::complex<data_type>, Eigen::Dynamic, Eigen::Dynamic> Matrix;
  Matrix A = aA;
  Matrix P = hessenberg_transformation(A, false);
  hessenberg_qr_iteration<Matrix>(A, false);
  Eigen::Matrix<data_type, Eigen::Dynamic, 1> res = A.diagonal().real();
  return res;
}

// TODO Check for template
template <typename Derived, class data_type = double> // First argument is Matrix type, second is type of the calculations
typename std::enable_if<std::is_arithmetic<typename Derived::Scalar>::value, Eigen::Matrix<data_type, Eigen::Dynamic, 1>>::type
qr_method(const Eigen::MatrixBase<Derived>& aA) {
  std::cout << "correct Method" << std::endl;
  const bool a_is_symmetric = is_symmetric(aA);  // Needs to be changed, or in the upper one
  if( a_is_symmetric ) return implicit_shift_qr_method<Derived, data_type>(aA);
  else return double_shift_qr_method<Derived, data_type>(aA);
}


#endif
