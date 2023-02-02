#ifndef HELPFUNCTIONS_HH_
#define HELPFUNCTIONS_HH_

// TODO maybe declaring inline functions static improves performance?

#include <type_traits>
#include <complex>
#include <iostream>
#include <fstream>
#include <vector>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#include <chrono>
#include <ctime>
#include <cassert>

template<class T> inline constexpr
std::enable_if_t<std::is_same<T, std::complex<typename T::value_type>>::value,
  bool> IsComplex() {
  return true;
}


template<class T> inline constexpr
std::enable_if_t<std::is_arithmetic<T>::value, bool>
IsComplex() {
  return false;
}


/* Determine if the given Matrix is hermitian
 * Parameter:
 * - ak_matrix: Matrix
 * Return: 'true', if ak_matrix is hermitian, 'false' else
 */
template <class Derived>
std::enable_if_t<!IsComplex<typename Derived::Scalar>(), bool>
IsHermitian(const Eigen::MatrixBase<Derived> &ak_matrix,
                  const double ak_tol = 1e-14) {
  for (int i = 0; i < ak_matrix.rows(); ++i) {
    for (int ii = 0; ii < ak_matrix.rows(); ++ii) {
      if (std::abs(ak_matrix(i, ii) - ak_matrix(ii, i)) >= ak_tol)
        return false;                                                             // The Matrix is not symmetric
    }
  }
  return true;
}

template <class Derived>
std::enable_if_t<IsComplex<typename Derived::Scalar>(), bool>
IsHermitian(const Eigen::MatrixBase<Derived> &ak_matrix,
                  const double ak_tol = 1e-14) {
  for (int i = 0; i < ak_matrix.rows(); ++i) {
    for (int ii = 0; ii < ak_matrix.rows(); ++ii) {
      if (std::abs(ak_matrix(i, ii) - std::conj(ak_matrix(ii, i))) >= ak_tol)
        return false;                                                             // The Matrix is not symmetric
    }
  }
  return true;
}


template<class T>
struct HasLesser{
  template<class U>
  static auto test(U*) -> decltype(std::declval<U>() < std::declval<U>());
  template<typename, typename>
  static auto test(...) -> std::false_type;

  template<class data>
  static constexpr bool check(){
    return std::is_same<bool, decltype(test<T, T>(0))>::value;
  }
};


template<class T> inline
std::enable_if_t<HasLesser<T>::check(), bool>
LesserEv(const T& c1, const T& c2) {
  return c1 < c2;
}


template<class T> inline
bool LesserEv(const std::complex<T>& ak_c1, const std::complex<T>& ak_c2,
    const double ak_tol = 1e-4) {
  if ((std::real(ak_c1) - std::real(ak_c2)) < - ak_tol) {
    return true;
  } else if (std::abs(std::real(ak_c1) - std::real(ak_c2)) <= ak_tol &&
      std::imag(ak_c1) < std::imag(ak_c2)) {
    return true;
  } else {
    return false;
  }
}


// Write an Eigen Matrix to and From CSV
template<typename Derived> inline
void saveData(const std::string& ak_filename,
    const Eigen::MatrixBase<Derived>& ak_mat)
{
	const static Eigen::IOFormat CSVFormat(Eigen::FullPrecision,
      Eigen::DontAlignCols, ", ", "\n");
  std::ofstream file(ak_filename);
	if (file.is_open())
	{
		file << ak_mat.format(CSVFormat);
		file.close();
	}
}


template<typename MatrixType>
MatrixType openData(const std::string& ak_file_to_open)
{
  std::vector<double> entries;
  std::ifstream DataFile(ak_file_to_open);
  std::string row_string;
  std::string entry;
	int number_of_rows = 0;

	while (std::getline(DataFile, row_string)) {
    std::stringstream matrixRowStringStream(row_string);                            //convert to stringstream
		while (std::getline(matrixRowStringStream, entry, ',')) {                       // Split the entries
			entries.push_back(stod(entry));
		}
		++number_of_rows;
	}
  if constexpr(MatrixType::IsRowMajor) {
    return Eigen::Map<MatrixType, 0, Eigen::InnerStride<1>>(entries.data(),
        number_of_rows, entries.size() / number_of_rows);
  } else {
    return Eigen::Map<MatrixType, 0, Eigen::OuterStride<1>>(entries.data(),
        number_of_rows, entries.size() / number_of_rows);
  }
}


template<class C>
std::ostream & operator<<(std::ostream& a_out, const std::vector<C>& ak_v)
{
  a_out << "{";
  for(auto iter = ak_v.begin(); iter != ak_v.end(); ++iter) {
    a_out << *iter;
    if(iter +1 != ak_v.end()) {
      a_out << ",";
    }
  }
  a_out << "}";
  return a_out;
}


template<typename Derived>
std::vector<typename Derived::value_type>
ConvertToVec(const Eigen::EigenBase<Derived>& ak_v) {
  typedef Eigen::Vector<typename Derived::value_type, -1> VectorType;
  std::vector<typename Derived::value_type> v2;
  v2.resize(ak_v.size());
  VectorType::Map(&v2[0], ak_v.size()) = ak_v;
  return v2;
}


// Creating Random Matrices
template<typename MatrixType>
MatrixType CreateUnitaryMatrix(const int ak_size,
    const int ak_seed = std::time(nullptr)) {
  std::srand(ak_seed);
  MatrixType A(ak_size, ak_size);
  A.setRandom();
  Eigen::HouseholderQR<MatrixType> qr(A);
  return qr.householderQ();
}


template<typename MatrixType>
MatrixType CreateStdRandom(const int ak_size,
    const int ak_seed = std::time(nullptr)) {
  std::srand(ak_seed);
  MatrixType A(ak_size, ak_size);
  for( auto& entry : A.reshaped()) {
      entry = (std::rand() % 200) - 100;
  }
  return A;
}


template<typename MatrixType>
MatrixType CreateStdRandomTridiagonal(const int ak_size,
                                const int ak_seed = std::time(nullptr)) {
  std::srand(ak_seed);
  MatrixType A = MatrixType::Zero(ak_size, ak_size);
  for (int i = 0; i < ak_size; ++i) {
    for (int j = i; j < ak_size; ++j) {
      A(i, j) = (std::rand() % 200) - 100;
    }
  }
  return A;
}


template<typename MatrixType>
MatrixType CreateStdRandomDiagonal(const int ak_size,
                                const int ak_seed = std::time(nullptr)) {
  std::srand(ak_seed);
  MatrixType A = MatrixType::Zero(ak_size, ak_size) ;
  for (int i = 0; i < ak_size; ++i) {
      int x = (std::rand() % 200) - 100;
    A(i, i) = x;
  }
  return A;
}


template<typename MatrixType>
MatrixType CreateStdRandomSchur(const int ak_size,
                                const int ak_seed = std::time(nullptr)) {
  std::srand(ak_seed);
  MatrixType A = CreateStdRandomTridiagonal<MatrixType>(ak_size, std::rand());
  for(int i = 0; i < ak_size-1; i+= 2) {
    A(Eigen::seqN(i, 2), Eigen::seqN(i, 2)) = CreateStdRandom<MatrixType>(2,
        std::rand());
  }
  return A;
}


template<typename MatrixType>
MatrixType CreateRandomDiagonal(const int ak_size,
                                     const int ak_seed = std::time(nullptr)) {
  std::srand(ak_seed);
  MatrixType A = MatrixType::Zero(ak_size, ak_size);
  MatrixType diag = MatrixType::Random(ak_size);
  A.diagonal() = diag;
  return A;
}


template <class MatrixType>
MatrixType CreateLaplaceMatrix(const int ak_size) {
  typedef Eigen::Triplet<int> T;
  std::vector<T> entries;
  for (int i = 0; i < ak_size; ++i) {
    for (int j = 0; j < ak_size; ++j) {
      entries.push_back(T(i * ak_size + j, i * ak_size + j, 4));
      if (i > 0) {
        entries.push_back(T(i * ak_size + j, (i - 1) * ak_size + j, -1));
      }
      if (i < ak_size - 1) {
        entries.push_back(T(i * ak_size + j, (i + 1) * ak_size + j, -1));
      }
      if (j > 0) {
        entries.push_back(T(i * ak_size + j, i * ak_size + j - 1, -1));
      }
      if (j < ak_size - 1) {
        entries.push_back(T(i * ak_size + j, i * ak_size + j + 1, -1));
      }
    }
  }
  MatrixType Matrix(ak_size * ak_size, ak_size * ak_size);
  Matrix.setFromTriplets(entries.begin(), entries.end());
  return Matrix;
}


template<typename MatrixType, typename T = std::complex<double>>
std::tuple<MatrixType, std::vector<T>>
CreateRandom(const int ak_size, const bool is_symm,
    const int ak_seed = std::time(nullptr)) {
  std::srand(ak_seed);
  MatrixType A;
  std::vector<T> res;
  res.reserve(ak_size);

  if( is_symm ) {
    A = CreateStdRandomDiagonal<MatrixType>(ak_size, ak_seed);
    std::vector<typename MatrixType::value_type> real_res =
      ConvertToVec(A.diagonal());
    for(auto& x : real_res) {
      res.push_back(T{x});
    }
  } else {
    res.resize(ak_size);
    A = CreateStdRandomSchur<MatrixType>(ak_size, ak_seed);
    for(int i = 0; i < ak_size-1; ++i) {
      T d = (A(i, i) + A(i + 1, i + 1)) / 2.0;
      T  pq = std::sqrt(T{ d * d - A(i, i) * A(i + 1, i + 1)
            + A(i, i + 1) * A(i + 1, i)});
      res.at(i) = d - pq;                                                         // First eigenvalue
      ++i;
      res.at(i) = d + pq;                                                         // Second eigenvalue
    }
    if(ak_size % 2 == 1) {
      res.at(ak_size - 1) = A(ak_size - 1, ak_size - 1);
    }
  }
  std::sort(res.begin(), res.end(), [](T& a, T& b) { return LesserEv(a, b);});
  MatrixType Q = CreateUnitaryMatrix<MatrixType>(ak_size, ak_seed);
  A = Q.adjoint() * A * Q;
  return std::tie(A, res);
}
#endif
