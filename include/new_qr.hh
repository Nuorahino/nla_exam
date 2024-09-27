#ifndef NEW_QR_HH_
#define NEW_QR_HH_

#include <eigen3/Eigen/Dense>

namespace nla_exam {

template <class Derived, class Derived2>
void
ApplyHouseholderLeft(const Eigen::MatrixBase<Derived2> &ak_w,
                     const Eigen::MatrixBase<Derived> &a_matrix,
                     const typename Derived::Scalar beta) {
  typedef typename Derived::Scalar T;
  typedef Eigen::MatrixBase<Derived> MatrixType;
  MatrixType &matrix = const_cast<MatrixType &>(a_matrix);  // Const cast needed for eigen

  for (int i = 0; i < a_matrix.cols(); ++i) {
    T tmp;
    if constexpr (IsComplex<T>()) {
      tmp = std::conj(beta) * ak_w.dot(a_matrix(Eigen::all, i));  // w.dot(A) = w.adjoint() * A
    } else {
      tmp = beta * (ak_w.dot(a_matrix(Eigen::all, i)));  // w.dot(A) = w.adjoint() * A
    }
    matrix(Eigen::all, i) -= tmp * ak_w;
  }
  return;
}


#ifdef ROWWISE
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
#else
template <class Derived, class Derived2>
void
ApplyHouseholderRight(const Eigen::MatrixBase<Derived2> &ak_w,
                      const Eigen::MatrixBase<Derived> &a_matrix,
                      const typename Derived::Scalar beta) {
  typedef typename Derived::Scalar T;
  typedef Eigen::MatrixBase<Derived> MatrixType;
  MatrixType &matrix = const_cast<MatrixType &>(a_matrix);  // Const cast needed for eigen
  int n = matrix.rows();

  Eigen::Vector<typename Derived::Scalar, -1> r_tmp = beta * matrix(Eigen::all, Eigen::all) * ak_w;
  for(int j = 0; j < matrix.cols(); ++j) {
    matrix(Eigen::all, j) -= r_tmp * ak_w.conjugate()(j);
  }
  return;
}
#endif


} // namespace nla_exam

#endif
