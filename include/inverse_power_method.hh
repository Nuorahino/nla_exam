#ifndef INVERSE_POWER_METHOD_HH_
#define INVERSE_POWER_METHOD_HH_

#include <iostream>

#include <eigen3/Eigen/Cholesky>
#include <eigen3/Eigen/Dense>

template <class DerivedMat, class DerivedVec, typename T = typename DerivedVec::Scalar>
T
InversePowerMethod(const Eigen::MatrixBase<DerivedMat>& ak_a,
                   const typename DerivedMat::Scalar ak_shift,
                   const Eigen::MatrixBase<DerivedVec>& a_v, const double ak_tol) {
  typedef Eigen::Matrix<typename DerivedMat::Scalar, -1, -1> MatrixType;
  MatrixType shifted_a = ak_a - MatrixType::Identity(ak_a.rows(), ak_a.cols()) * 4;
  Eigen::LDLT dec = shifted_a.ldlt();  // Construct LU decomposition to solve linalg
  Eigen::MatrixBase<DerivedVec>& v = const_cast<Eigen::MatrixBase<DerivedVec>&>(a_v);
  double alpha = 0;
  {
    double step;
    double new_alpha;
    do {
      v = dec.solve(v);  // solve for Ay = v
      v.normalize();
      new_alpha = (ak_a * v).dot(v) / v.squaredNorm();
      step = std::abs(alpha - new_alpha);
      alpha = new_alpha;
    } while (step > ak_tol);
  }
  return alpha;
}

#endif  // INVERSE_POWER_METHOD_HH_
