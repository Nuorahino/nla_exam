#ifndef SYMM_QR_HH_
#define SYMM_QR_HH_

#include <type_traits>
#include <cmath> // for std::copysign
#include <vector>
#include <iostream>

#ifdef USELAPACK
#include "../lapack/lapack_interface_impl.hh"
#endif

#include "helpfunctions.hh"
#include "new_qr.hh"


namespace nla_exam {
/* Compute the Householder Vector
 * Parameter:
 * - ak_x: Vector to transform to multiple of e1
 * - a_tol: double, under which the tail is considered 0
 * Return: Householder Vector
 */
template <class Derived, class T = typename Derived::Scalar>
Eigen::Vector<T, -1>
GetHouseholderVector(const Eigen::MatrixBase<Derived> &ak_x) {
  typedef typename RealType<IsComplex<T>(), T>::type RT;
  Eigen::Vector<T, -1> w = ak_x;
  int64_t n = w.rows();
  T beta = w(Eigen::lastN(n - 1)).squaredNorm();
  if (std::abs(beta) < std::numeric_limits<RT>::min()) {
  //if (beta < std::numeric_limits<T>::min()) {
    w(0) = 1;
  } else {
    T s = std::sqrt(std::abs(w(0)) * std::abs(w(0)) + beta);
    if constexpr (IsComplex<typename Derived::Scalar>()) {
      s *= w(0) / std::abs(w(0));  // Choise to avoid loss of significance
    } else {
      if (w(0) < 0) s *= -1;
    }
    w(0) = w(0) + s;
  }
  return w;
}

#ifdef OLD_PARAM
template <class Derived, class T = typename Derived::Scalar>
Eigen::Vector<T, -1>
GetHouseholderVector(const Eigen::MatrixBase<Derived> &ak_x, T &beta) {
  typedef typename RealType<IsComplex<T>(), T>::type RT;
  Eigen::Vector<T, -1> w = ak_x;
  int64_t n = w.rows();
  beta = w(Eigen::lastN(n - 1)).squaredNorm();
  // TODO (Georg): Better Criteria needed
  if (std::abs(beta) < std::numeric_limits<RT>::min()) {
    w(0) = 1;
  } else {
    T s = std::sqrt(std::abs(w(0)) * std::abs(w(0)) + beta);
    if constexpr (IsComplex<T>()) {
      s *= w(0) / std::abs(w(0));  // Choise to avoid loss of significance
      // s *= std::polar(1.0, std::arg(w(0)));  // Does not work for float and double at
      // the same time
    } else {
      if (w(0) < 0) s *= -1;
    }
    w(0) = w(0) + s;
  }
  beta += std::abs(w(0) * w(0));
  beta = T{2} / beta;
  return w;
}
#else
template <class Derived, class T = typename Derived::Scalar>
Eigen::Vector<T, -1>
GetHouseholderVector(const Eigen::MatrixBase<Derived> &ak_x, T &beta) {
  typedef typename RealType<IsComplex<T>(), T>::type RT;
  Eigen::Vector<T, -1> w = ak_x;
  int64_t n = w.rows();
  beta = w(Eigen::lastN(n - 1)).squaredNorm();
  // TODO (Georg): Better Criteria needed
  if constexpr (IsComplex<T>()) {
    T norm = std::sqrt(std::abs(w(0)) * std::abs(w(0)) + beta);
    T eta = w(0).real() / std::abs(w(0).real()) * norm;
    w(0) += eta;
    beta = w(0) / eta;
    w /= w(0);
  } else {
    if (std::abs(beta) < std::numeric_limits<RT>::min()) {
      w(0) = 1;
      beta = 0;
    } else {
    T norm = std::sqrt(std::abs(w(0)) * std::abs(w(0)) + beta);
      T eta = w(0) / std::abs(w(0)) * norm;
      w(0) += eta;
      beta = w(0) / eta;
      w /= w(0);
    }
  }
  return w;
}
#endif



template <class DataType, bool is_symmetric, class Matrix>
inline std::enable_if_t<is_symmetric, std::vector<DataType>>
CalcEigenvaluesFromSchur(const Matrix &ak_matrix,
                         [[maybe_unused]] const bool ak_matrix_is_diagonal = false) {
  std::vector<DataType> res(rows<Matrix>(ak_matrix));
  for (unsigned i = 0; i < rows<Matrix>(ak_matrix); ++i) {
    res.at(i) = ak_matrix(i, i);
  }
  return res;
}

template <bool hermitian, bool colmajor, class Derived>
std::enable_if_t<hermitian && colmajor, void>
HessenbergTransformation(const Eigen::MatrixBase<Derived> &a_matrix) {
  typedef Eigen::MatrixBase<Derived> MatrixType;
  MatrixType &matrix = const_cast<MatrixType &>(a_matrix);
  int64_t n = a_matrix.rows();

  typename Derived::Scalar beta;
  //for (int i = 0; i < matrix.rows() - 2; ++i) {
  for (int i = 0; i < matrix.rows() - 1; ++i) {
    Eigen::Vector<typename Derived::Scalar, -1> w =
        GetHouseholderVector(matrix(Eigen::lastN(n - i - 1), i), beta);

#ifdef VERSION1
    ApplyHouseholderLeft(w, matrix(Eigen::lastN(n - i - 1), Eigen::lastN(n - i)), beta);
    ApplyHouseholderRight(w, matrix(Eigen::lastN(n - i), Eigen::lastN(n - i - 1)), beta);

    // Version 2
#else
#ifdef VERSION2
    // Apply Left
    for (int j = i; j < n; ++j) {
    typename Derived::Scalar tmp;
    if constexpr (IsComplex<typename Derived::Scalar>()) {
        tmp = std::conj(beta) * w.dot(a_matrix(Eigen::lastN(n - i - 1), j));  // w.dot(A) = w.adjoint() * A
      } else {
        tmp = beta * w.dot(a_matrix(Eigen::lastN(n - i - 1), j));  // w.dot(A) = w.adjoint() * A
      }
      matrix(Eigen::lastN(n - i - 1), j) -= (tmp * w).eval();
    }
    // Apply Right
    Eigen::Vector<typename Derived::Scalar, -1> r_tmp = beta * matrix(Eigen::lastN(n - i), Eigen::lastN(n - i - 1)) * w;
    for(int j = 0; j < n - i - 1; ++j) {
      matrix(Eigen::lastN(n - i), i + j + 1) -= r_tmp * w.adjoint()(j);
    }

#else
#ifdef VERSION4
 //Version 4:
Eigen::Vector<typename Derived::Scalar, -1> tmp(n - i);
    //Eigen::Vector<typename Derived::Scalar, -1> tmp;
    if constexpr (IsComplex<typename Derived::Scalar>()) {
      tmp = std::conj(beta) * w.adjoint() * (a_matrix(Eigen::lastN(n - i - 1), Eigen::lastN(n - i)));
    } else {
      tmp = beta * w.transpose() * (a_matrix(Eigen::lastN(n - i - 1), Eigen::lastN(n - i)));
    }

    typename Derived::Scalar test;
    if constexpr (IsComplex<typename Derived::Scalar>()) {
      test = tmp(Eigen::lastN(n - i - 1)).conjugate().dot(w) * beta;
    } else {
      test = tmp(Eigen::lastN(n - i - 1)).dot(w) * beta;
    }
    // Apply both
    Eigen::Vector<typename Derived::Scalar, -1> r_tmp = tmp.conjugate();
    r_tmp(Eigen::lastN(n - i - 1)) -= test * w;
    matrix(Eigen::lastN(n - i - 1), i) -= (tmp(0) * w).eval();
    for(int j = 0; j < n - i - 1; ++j) {
      matrix(Eigen::lastN(n - i), i + j + 1) -= r_tmp * w.adjoint()(j);
      matrix(Eigen::lastN(n - i - 1), i + j + 1) -= (tmp(j + 1) * w).eval();
    }

#else

 //Version 3:
    Eigen::Vector<typename Derived::Scalar, -1> tmp(n - i);
    // Apply Left
    for (int j = i; j < n; ++j) {
    if constexpr (IsComplex<typename Derived::Scalar>()) {
        tmp(j - i) = std::conj(beta) * w.dot(a_matrix(Eigen::lastN(n - i - 1), j));  // w.dot(A) = w.adjoint() * A
      } else {
        tmp(j - i) = beta * w.dot(a_matrix(Eigen::lastN(n - i - 1), j));  // w.dot(A) = w.adjoint() * A
      }
      matrix(Eigen::lastN(n - i - 1), j) -= (tmp(j - i) * w).eval();
    }

    typename Derived::Scalar test;
    if constexpr (IsComplex<typename Derived::Scalar>()) {
      test = tmp(Eigen::lastN(n - i - 1)).conjugate().dot(w) * beta;
    } else {
      test = tmp(Eigen::lastN(n - i - 1)).dot(w) * beta;
    }
    // Apply Right
    //r_tmp(Eigen::lastN(n - i - 1)) = tmp(Eigen::lastN(n - i - 1)).conjugate() - test * w;
    Eigen::Vector<typename Derived::Scalar, -1> r_tmp = tmp.conjugate();
    r_tmp(Eigen::lastN(n - i - 1)) -= test * w;
    for(int j = 0; j < n - i - 1; ++j) {
      matrix(Eigen::lastN(n - i), i + j + 1) -= r_tmp * w.adjoint()(j);
    }
#endif
#endif
#endif



    matrix(Eigen::seqN(i + 2, n - i - 2), i) = MatrixType::Zero(n - i - 2, 1);
    matrix(i, Eigen::seqN(i + 2, n - i - 2)) = MatrixType::Zero(1, n - i - 2);
  }

//  if constexpr (IsComplex<typename Derived::Scalar>()) {
//    // Transform complex Hermitian Matrix to a Real Tridiagonal Matrix
//    for (int i = 1; i < a_matrix.rows(); ++i) {
//      matrix(i - 1, i) = std::abs(a_matrix(i - 1, i));
//      matrix(i, i - 1) = std::abs(a_matrix(i, i - 1));
//    }
//  }
  return;
}

/* Get Wilkinson shift parameter for a given 2x2 Matrix
 * Parameter:
 * - ak_matrix: 2x2 Matrix of which to calculate the shift parameter
 * Return: value of the shift parameter
 */
template <class DataType, bool is_symmetric, class Matrix>
//inline std::enable_if_t<is_symmetric && std::is_arithmetic<DataType>::value, DataType>
std::enable_if_t<is_symmetric && std::is_arithmetic<DataType>::value, DataType>
WilkinsonShift(const Matrix &ak_matrix, const int i) {
  DataType d = (ak_matrix(i, i) - ak_matrix(i + 1, i + 1)) / static_cast<DataType>(2.0);
  if (d >= 0) {
    return ak_matrix(i + 1, i + 1) + d - std::hypot(d, ak_matrix(i + 1, i));
  } else {
    return ak_matrix(i + 1, i + 1) + d + std::hypot(d, ak_matrix(i + 1, i));
  }
}


/* Deflates a Matrix converging to a diagonal matrix
 * Parameter:
 * - a_matrix: Matrix to deflate
 * - a_begin: Index of fhe block that is solved currently
 * - a_end: Index of fhe End that is solved currently
 * - ak_tol: Tolerance for considering a value 0
 * Return: "true" if the block is fully solved, "false" otherwise
 */
template <class Matrix>
inline int
DeflateDiagonal(Matrix &a_matrix, int &a_begin, int &a_end,
                const double ak_tol = 1e-12) {
  int state = 2;
  for (int i = a_end; i > a_begin; --i) {
    if (std::abs(a_matrix(i, i - 1)) <=
        (ak_tol * (std::abs(a_matrix(i, i)) + std::abs(a_matrix(i - 1, i - 1))))) {
      a_matrix(i, i - 1) = 0;
      if (state < 2) {
        a_begin = i;
        return 1;  // Subblock to solve found
      }
    } else if (state == 2) {  // Start of the block found
      if (i == a_end) {
        state = 0;
      } else {
        a_end = i;
        state = 1;
      }
    }
  }
  return state;
}


/* Calculate the entries of a Givens Matrix
 * Parameter:
 * - ak_a: first entry
 * - ak_b: entry to eliminate
 * Return: Vector containing {c, s}
 */
#ifdef USELAPACK
template <class DataType>
std::enable_if_t<std::is_arithmetic<DataType>::value, void>
GetGivensEntries(const DataType &ak_a, const DataType &ak_b, std::array<DataType, 3> &entries) {
  return compute_givens_parameter(ak_a, ak_b, entries);
}
#else
template <class DataType>
std::enable_if_t<std::is_arithmetic<DataType>::value, void>
GetGivensEntries(const DataType &ak_a, const DataType &ak_b, std::array<DataType, 3> &entries) {
  if (std::abs(ak_a) <= std::numeric_limits<DataType>::epsilon()) {
    entries.at(0) = 0;
    entries.at(1) = 1;
    entries.at(2) = ak_a;
  } else {
    DataType r = std::hypot(ak_a, ak_b);
    entries.at(0) = std::abs(ak_a) / r;
    entries.at(2) = std::copysign(r, ak_a);
    entries.at(1) = ak_b / entries.at(2);
  }
  return;
}
#endif


template <bool first, bool last, bool is_symmetric,
  class DataType, class Matrix>
inline std::enable_if_t<std::is_arithmetic<DataType>::value && is_symmetric, void>
ApplyGivensTransformation(Matrix &matrix, const std::array<DataType, 3> &entries, const int aBegin, DataType &buldge) {
    int k = aBegin;
    DataType ak_c = entries.at(0);
    DataType ak_s = entries.at(1);
    DataType ss = ak_s * ak_s;
    DataType cs = ak_c * ak_s;
    DataType cc = ak_c * ak_c;
    DataType csn = cs * (2 * matrix(k+1, k));

    matrix(k+1, k) = (cc - ss) * matrix(k+1, k) + cs * (matrix(k+1, k+1) - matrix(k,k)) ;
    DataType x1 = cc * matrix(k, k) + ss * matrix(k+1,k+1) + csn;
    matrix(k+1, k+1) = cc * matrix(k+1, k+1) + ss * matrix(k,k) - csn;
    matrix(k, k) = x1;

if constexpr (!first) {
  matrix(k, k - 1) = entries.at(2);
}

if constexpr (!last) {
    buldge = ak_s * matrix(k+2, k+1);
    matrix(k+2, k+1) *= ak_c;
}


  return;
}


/* Executes one step of the implicit qr algorithm for a tridiagonal Matrix
 * Parameter:
 * - a_matrix: Tridiagonal Matrix
 * Return: void
 */
template <class DataType, bool is_symmetric, typename Matrix>
std::enable_if_t<std::is_arithmetic<DataType>::value && is_symmetric, void>
ImplicitQrStep(Matrix &matrix,
               const int aBegin, const int aEnd) {
  int n = aEnd - aBegin + 1;
  DataType shift = WilkinsonShift<DataType, is_symmetric>(
        matrix, aEnd - 1);
  std::array<typename ElementType<Matrix>::type, 3> entries;
  GetGivensEntries<>(matrix(aBegin, aBegin) - shift,
                                    matrix(aBegin + 1, aBegin), entries);  // Parameter for the initial step
  // innitial step
  DataType buldge = 0;
  if (n == 2) { //
    ApplyGivensTransformation<true, true, true, DataType>(matrix, entries, aBegin, buldge);
    return;
  }
  ApplyGivensTransformation<true, false, true, DataType>(matrix, entries, aBegin, buldge);
  // buldge chasing
  for (int k = aBegin + 1; k < aEnd - 1; ++k) {
    GetGivensEntries<>(matrix(k, k - 1), buldge, entries);
    ApplyGivensTransformation<false, false, true, DataType>(matrix, entries, k, buldge); // this one is not inline
  }
  GetGivensEntries<>(matrix(aEnd - 1, aEnd - 2), buldge, entries);
  ApplyGivensTransformation<false, true, true, DataType>(matrix, entries, aEnd - 1, buldge);

  return;
}


/* get the eigenvalues of a hessenberg matrix using the qr iteration
 * parameter:
 * - a_matrix: hessenberg matrix
 * - ak_is_hermitian: "true" if a_matrix is symmetric, "false" otherwise
 * - ak_tol: tolerance for considering a value 0
 * return: unordered vector of eigenvalues
 */
template <class DataType, bool is_hermitian, class matrix>
std::enable_if_t<std::is_arithmetic<typename ElementType<matrix>::type>::value && is_hermitian, std::vector<DataType>>
QrIterationHessenberg(matrix &a_matrix,
                      const double ak_tol = 1e-12) {
  // generell definitions
  int begin = 0;
  int end = rows<matrix>(a_matrix) - 1;

  // qr iteration
  while (0 < end) {
    int status = DeflateDiagonal<matrix>(a_matrix, begin, end, ak_tol);
    if (status > 1) {
      end = begin - 1;
      begin = 0;
    } else {
      ImplicitQrStep<typename ElementType<matrix>::type, is_hermitian, matrix>(a_matrix, begin, end);
    }
  }
  return CalcEigenvaluesFromSchur<DataType, true>(a_matrix, true);
}

template <
    bool IsHermitian, typename Matrix,
    class DataType = typename ElementType<Matrix>::type,
    class ComplexDataType = typename EvType<IsComplex<DataType>(), DataType>::type>
inline std::enable_if_t<IsHermitian, std::vector<ComplexDataType>>
QrMethod(const Matrix &ak_matrix, const double ak_tol = 1e-12) {
  static_assert(std::is_convertible<typename ElementType<Matrix>::type, DataType>::value,
                "Matrix Elements must be convertible to DataType");
  Matrix A = ak_matrix;  // Do not change the input matrix
  return QrIterationHessenberg<ComplexDataType, IsHermitian>(A, ak_tol);
}

} // namespace nla_exam

#endif
