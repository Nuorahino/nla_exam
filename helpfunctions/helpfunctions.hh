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
std::enable_if_t<std::is_same<T , std::complex<typename T::value_type>>::value, bool>
is_complex() {
  return true;
}

template<class T> inline constexpr
std::enable_if_t<std::is_arithmetic<T>::value, bool>
is_complex() {
  return false;
}

template<class T>
struct has_lesser {
  template<class U>
  static auto test(U*) -> decltype(std::declval<U>() < std::declval<U>());
  template<typename, typename>
  static auto test(...) -> std::false_type;


  template<class data>
  static constexpr bool check(){
    return std::is_same<bool, decltype(test<T, T>(0))>::value;
  }
};

 template <typename T> inline constexpr
 double signum(const T x, std::false_type) {
     return T{0} <= x;
 }

 template <typename T> inline constexpr
 double signum(const T x, std::true_type) {
     return (T{0} <= x) - (x < T{0});
 }

 template <typename T> inline constexpr
 double signum(const T x) {
     return signum(x, std::is_signed<T>());
 }

 template <typename T> inline constexpr
 double signum(const std::complex<T> x) {
     return signum(x.real(), std::is_signed<T>());
 }

template<class T> inline
std::enable_if_t<has_lesser<T>::check(), bool>
lesser_ev(const T& c1, const T& c2) {
  return c1 < c2;
}

template<class T> inline
bool lesser_ev(const std::complex<T>& c1, const std::complex<T>& c2, const double aTol = 1e-4) {
  if ((std::real(c1) - std::real(c2)) < - aTol) {
    return true;
  } else if (std::abs(std::real(c1) - std::real(c2)) <= aTol && std::imag(c1) < std::imag(c2)) {
    return true;
  } else {
    return false;
  }
}


// Write an Eigen Matrix to and From CSV
template<typename Derived> inline
void saveData(const std::string& fileName, const Eigen::MatrixBase<Derived>&  matrix)
{
	const static Eigen::IOFormat CSVFormat(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
  std::ofstream file(fileName);
	if (file.is_open())
	{
		file << matrix.format(CSVFormat);
		file.close();
	}
}

template<typename Matrix>
Matrix openData(const std::string& fileToOpen)
{
  std::vector<double> matrixEntries;
  std::ifstream DataFile(fileToOpen);
  std::string matrixRowString;
  std::string matrixEntry;
	int number_of_rows = 0;

	while (std::getline(DataFile, matrixRowString)) // Read the file in rowwise
	{
    std::stringstream matrixRowStringStream(matrixRowString); //convert to stream

		while (std::getline(matrixRowStringStream, matrixEntry, ',')) // split of the entries
		{
			matrixEntries.push_back(stod(matrixEntry));   // convert string to double
		}

		++number_of_rows; //update the column numbers
	}
  if constexpr(Matrix::IsRowMajor) {
    return Eigen::Map<Matrix, 0, Eigen::InnerStride<1>>(matrixEntries.data(), number_of_rows, matrixEntries.size() / number_of_rows);
  } else {
    return Eigen::Map<Matrix, 0, Eigen::OuterStride<1>>(matrixEntries.data(), number_of_rows, matrixEntries.size() / number_of_rows);
  }
}

template<class C>
std::ostream & operator<<(std::ostream& out, const std::vector<C>& v)
{
  out << "{";
  size_t last = v.size() - 1;
  for(size_t i = 0; i < v.size(); ++i) {
    out << v[i];
    if (i != last)
      out << ", ";
  }
  out << "}";
  return out;
}

// Function for converting Eigen to std Vector
template<typename Derived>
std::vector<typename Derived::value_type> convertToVec(const Eigen::EigenBase<Derived>& aV) {
  std::vector<typename Derived::value_type> v2;
  typedef Eigen::Vector<typename Derived::value_type, -1> Vec;
  v2.resize(aV.size());
  Vec::Map(&v2[0], aV.size()) = aV;
  return v2;
}

// Creating Random Matrices
template<typename Matrix>
Matrix CreateUnitaryMatrix(const int aN,
    const int aSeed = std::time(nullptr)) {
  std::srand(aSeed);
  Matrix A(aN, aN);
  A.setRandom();
  Eigen::HouseholderQR<Matrix> qr(A);
  return qr.householderQ();
}


template<typename Matrix>
Matrix CreateStdRandom(const int aN, const int aSeed = std::time(nullptr)) {
  std::srand(aSeed);
  Matrix A(aN, aN);
  for (int i = 0; i < aN; ++i) {
    for (int j = 0; j < aN; ++j) {
      int x = (std::rand() % 200) - 100;
      A(i, j) = x;
    }
  }
  return A;
}

template<typename Matrix>
Matrix CreateStdRandomTridiagonal(const int aN,
                                const int aSeed = std::time(nullptr)) {
  std::srand(aSeed);
  Matrix A = Matrix::Zero(aN, aN);
  for (int i = 0; i < aN; ++i) {
    for (int j = i; j < aN; ++j) {
      int x = (std::rand() % 200) - 100;
      A(i, j) = x;
    }
  }
  return A;
}

template<typename Matrix>
Matrix CreateStdRandomDiagonal(const int aN,
                                const int aSeed = std::time(nullptr)) {
  std::srand(aSeed);
  Matrix A = Matrix::Zero(aN, aN) ;
  for (int i = 0; i < aN; ++i) {
      int x = (std::rand() % 200) - 100;
    A(i, i) = x;
  }
  return A;
}

template<typename Matrix>
Matrix CreateStdRandomSchur(const int aN,
                                const int aSeed = std::time(nullptr)) {
  std::srand(aSeed);
  Matrix A = CreateStdRandomTridiagonal<Matrix>(aN, std::rand());
  for(int i = 0; i < aN-1; i+= 2) {
    A(Eigen::seqN(i, 2), Eigen::seqN(i, 2)) = CreateStdRandom<Matrix>(2, std::rand());
  }
  return A;
}


template<typename Matrix>
Matrix CreateRandomDiagonal(const int aN,
                                     const int aSeed = std::time(nullptr)) {
  std::srand(aSeed);
  Matrix A = Matrix::Zero(aN, aN);
  Matrix diag = Matrix::Random(aN);
  A.diagonal() = diag;
  return A;
}

template <class Mat>
Mat CreateLaplaceMatrix(int n) {
  typedef Eigen::Triplet<int> T;
  std::vector<T> entries;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      entries.push_back(T(i * n + j, i * n + j, 4));
      if (i > 0) entries.push_back(T(i * n + j, (i - 1) * n + j, -1));
      if (i < n - 1) entries.push_back(T(i * n + j, (i + 1) * n + j, -1));
      if (j > 0) entries.push_back(T(i * n + j, i * n + j - 1, -1));
      if (j < n - 1) entries.push_back(T(i * n + j, i * n + j + 1, -1));
    }
  }
  Mat Matrix(n * n, n * n);
  Matrix.setFromTriplets(entries.begin(), entries.end());

  return Matrix;
}


template<typename Matrix, typename T = std::complex<double>>
std::tuple<Matrix, std::vector<T>> CreateRandom(const int aN, const bool is_symm,
    const int aSeed = std::time(nullptr)) {
  std::srand(aSeed);
  Matrix A;
  std::vector<T> res;
  res.reserve(aN);
  if( is_symm ) {
    A = CreateStdRandomDiagonal<Matrix>(aN, aSeed);
    std::vector<typename Matrix::value_type> real_res = convertToVec(A.diagonal());
    for(auto& x : real_res) {
      res.push_back(T{x});
    }
  } else {
    res.resize(aN);
    A = CreateStdRandomSchur<Matrix>(aN, aSeed);
    for(int i = 0; i < aN-1; ++i) {
      T d = (A(i, i) + A(i + 1, i + 1)) / 2.0;
      T  pq = std::sqrt(T{ d * d - A(i, i) * A(i + 1, i + 1)
            + A(i, i + 1) * A(i + 1, i)});
      res.at(i) = d - pq;                                                     // First eigenvalue
      ++i;
      res.at(i) = d + pq;                                                     // Second eigenvalue
    }
    if(aN % 2 == 1) {
      res.at(aN - 1) = A(aN - 1, aN - 1);
    }
  }
  std::sort(res.begin(), res.end(), [](T& a, T& b) { return lesser_ev(a, b);});
  Matrix Q = CreateUnitaryMatrix<Matrix>(aN, aSeed);
  A = Q.adjoint() * A * Q;
  return std::tie(A, res);
}
#endif
