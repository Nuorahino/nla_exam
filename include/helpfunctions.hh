#ifndef HELPFUNCTIONS_HH_
#define HELPFUNCTIONS_HH_

// TODO (Georg): maybe declaring inline functions static improves performance?

#include <cassert>
#include <chrono>
#include <complex>
#include <ctime>
#include <fstream>
#include <iostream>
#include <type_traits>
#include <vector>

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>

// TODO (Georg): add a namespace around this file
// TODO (Georg): Maybe rename to is_std_complex?
template <typename>
struct is_complex : std::false_type {};

template <typename T>
struct is_complex<std::complex<T>> : std::true_type {};

template <class T>
inline constexpr bool
IsComplex() {
  return is_complex<T>::value;
}

template <bool Complex, class T>
struct EvType {
  typedef T type;
};

template <class T>
struct EvType<false, T> {
  typedef std::complex<T> type;
};

template <bool Complex>
struct DoubleType {
  typedef std::complex<double> type;
};

template <>
struct DoubleType<false> {
  typedef double type;
};

/* Determine if the given Matrix is hermitian
 * Parameter:
 * - ak_matrix: Matrix
 * Return: 'true', if ak_matrix is hermitian, 'false' else
 */
// TODO (12.11): What to do in case of a matrix of size 0
template <class Derived>
std::enable_if_t<!IsComplex<typename Derived::Scalar>(), bool>
IsHermitian(const Eigen::MatrixBase<Derived>& ak_matrix, const double ak_tol = 1e-14) {
  if (ak_matrix.rows() != ak_matrix.cols()) {
    return false;
  }
  for (int i = 0; i < ak_matrix.rows(); ++i) {
    for (int ii = 0; ii < ak_matrix.rows(); ++ii) {
      if (std::abs(ak_matrix(i, ii) - ak_matrix(ii, i)) >= ak_tol) {
        return false;  // The Matrix is not symmetric
      }
    }
  }
  return true;
}

template <class Derived>
std::enable_if_t<IsComplex<typename Derived::Scalar>(), bool>
IsHermitian(const Eigen::MatrixBase<Derived>& ak_matrix, const double ak_tol = 1e-14) {
  if (ak_matrix.rows() != ak_matrix.cols()) {
    return false;
  }
  for (int i = 0; i < ak_matrix.rows(); ++i) {
    for (int ii = 0; ii < ak_matrix.rows(); ++ii) {
      if (std::abs(ak_matrix(i, ii) - std::conj(ak_matrix(ii, i))) >= ak_tol) {
        return false;  // The Matrix is not symmetric
      }
    }
  }
  return true;
}

template <class T>
struct HasLesser {
  template <class U>
  static auto test(U*) -> decltype(std::declval<U>() < std::declval<U>());
  template <typename, typename>
  static auto test(...) -> std::false_type;

  template <class data>
  static constexpr bool check() {
    return std::is_same<bool, decltype(test<T, T>(0))>::value;
  }
};

template <class T>
inline std::enable_if_t<HasLesser<T>::check(), bool>
LesserEv(const T& c1, const T& c2) {
  return c1 < c2;
}

template <class T>
inline bool
LesserEv(const std::complex<T>& ak_c1, const std::complex<T>& ak_c2,
         const double ak_tol = 1e-4) {
  if (((std::real(ak_c1) - std::real(ak_c2)) < -ak_tol) ||
      (std::abs(std::real(ak_c1) - std::real(ak_c2)) <= ak_tol &&
       std::imag(ak_c1) < std::imag(ak_c2))) {
    return true;
  } else {
    return false;
  }
}

// Write an Eigen Matrix to and From CSV
template <typename Derived>
inline void
saveData(const std::string& ak_filename, const Eigen::MatrixBase<Derived>& ak_mat) {
  const static Eigen::IOFormat CSVFormat(Eigen::FullPrecision, Eigen::DontAlignCols, ", ",
                                         "\n");
  std::ofstream file(ak_filename);
  if (file.is_open()) {
    file << ak_mat.format(CSVFormat);
    file.close();
  }
}

template <typename MatrixType>
MatrixType
openData(const std::string& ak_file_to_open) {
  std::vector<double> entries;
  std::ifstream DataFile(ak_file_to_open);
  std::string row_string;
  std::string entry;
  int number_of_rows = 0;

  while (std::getline(DataFile, row_string)) {
    std::stringstream matrixRowStringStream(row_string);       // convert to stringstream
    while (std::getline(matrixRowStringStream, entry, ',')) {  // Split the entries
      entries.push_back(stod(entry));
    }
    ++number_of_rows;
  }
  if constexpr (MatrixType::IsRowMajor) {
    return Eigen::Map<MatrixType, 0, Eigen::InnerStride<1>>(
        entries.data(), number_of_rows, entries.size() / number_of_rows);
  } else {
    return Eigen::Map<MatrixType, 0, Eigen::OuterStride<1>>(
        entries.data(), number_of_rows, entries.size() / number_of_rows);
  }
}

template <class C>
std::ostream&
operator<<(std::ostream& a_out, const std::vector<C>& ak_v) {
  a_out << "{";
  for (auto iter = ak_v.begin(); iter != ak_v.end(); ++iter) {
    a_out << *iter;
    if (iter + 1 != ak_v.end()) {
      a_out << ",";
    }
  }
  a_out << "}";
  return a_out;
}

template <typename Derived>
std::vector<typename Derived::value_type>
ConvertToVec(const Eigen::EigenBase<Derived>& ak_v) {
  typedef Eigen::Vector<typename Derived::value_type, -1> VectorType;
  std::vector<typename Derived::value_type> v2;
  v2.resize(ak_v.size());
  VectorType::Map(&v2[0], ak_v.size()) = ak_v;
  return v2;
}
#endif
