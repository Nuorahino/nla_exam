#ifndef TEST_HOUSEHOLDER_HH
#define TEST_HOUSEHOLDER_HH

#include <eigen3/Eigen/Dense>
#include <complex>


template<class Derived, class Derived2>
void ApplyHouseholder(const Eigen::MatrixBase<Derived2> &ak_x,
                      const Eigen::MatrixBase<Derived> &a_matrix,
                      const long a_start,
                      const long a_start_row,
                      const double ak_tol = 1e-12,
                      const std::complex<double> alphaMod = 1) {
  typedef typename Derived::Scalar T;
  typedef Eigen::MatrixBase<Derived> MatrixType;
  MatrixType& matrix = const_cast<MatrixType&>(a_matrix);
  Eigen::Vector<T, -1> w = ak_x;
  long n = w.rows();
  T alpha = w.norm();
  //if constexpr (IsComplex<typename Derived::Scalar>()) {
    alpha *= std::polar(1.0, arg(w(0)));                                       // Choise to avoid loss of significance
//  } else {
//    if (w(0) < 0) alpha *= -1;
//  }
  alpha *= alphaMod;                                                            // Test if this breaks the zeros
  w(0) = ak_x(0) + alpha;
  if (w.squaredNorm() < ak_tol) return;
  T beta = 2 / w.squaredNorm();
  for(int i = a_start; i < a_matrix.cols(); ++i) {
    alpha = beta * w.dot(a_matrix(Eigen::seqN(a_start_row, n), i));
    matrix(Eigen::seqN(a_start_row, n),i) -= alpha * w;
  }
  for(int i = 0; i < a_matrix.rows(); ++i) {
    alpha = beta * a_matrix(i, Eigen::seqN(a_start_row, n)) * w;
    matrix(i, Eigen::seqN(a_start_row, n)) -= alpha * w.adjoint().eval();
  }
}

#endif
