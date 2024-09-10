#ifndef CREATEMATRIX_HH_
#define CREATEMATRIX_HH_

#include <vector>
#include <complex>
#include <random>
#include <eigen3/Eigen/Dense>

#include "helpfunctions.hh"

template<bool Complex>
struct IntType {typedef std::complex<int> type;};

template<>
struct IntType<false> {typedef int type;};



//// Creating Random Matrices
//template<typename MatrixType>
//MatrixType CreateUnitaryMatrix(const int ak_size,
//    const int ak_seed = std::time(nullptr)) {
//  std::srand(ak_seed);
//  MatrixType A(ak_size, ak_size);
//  A.setRandom();
//  Eigen::HouseholderQR<MatrixType> qr(A);
//  return qr.householderQ();
//}

template<typename Vector>
void
CreateStdRandomVector(Vector &vec, const int ak_seed = std::time(nullptr)) {
  //std::srand(ak_seed);
  std::mt19937 rng(ak_seed);
  std::uniform_real_distribution<double> distribution(-1000, 1000);
  std::size_t n = vec.size();
  for (int i = 0; i < n; ++i) {
    vec.at(i) = distribution(rng);
  }
  return;
}
template<typename Matrix>
void
CreateStdRandomTri(Matrix &mat, const int ak_seed = std::time(nullptr)) {
  //std::srand(ak_seed);
  std::mt19937 rng(ak_seed);
  std::uniform_real_distribution<double> distribution(-1000, 1000);
  std::size_t n = mat.rows();
  for (int i = 0; i < n - 1; ++i) {
    mat(i, i) = distribution(rng);
    mat(i, i + 1) = distribution(rng);
    mat(i + 1, i) = mat(i, i + 1);
  }
  mat(n - 1, n - 1) = distribution(rng);
  return;
}


template<typename MatrixType>
MatrixType CreateStdRandom(const int ak_size,
    const int ak_seed = std::time(nullptr)) {
  std::srand(ak_seed);
  MatrixType A(ak_size, ak_size);
  for (auto& entry : A.reshaped()) {
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
MatrixType
CreateStdRandomDiagonal(const int ak_size,
                                const int ak_seed = std::time(nullptr)) {
  //std::srand(ak_seed);
  std::mt19937 rng(ak_seed);
  std::uniform_real_distribution<double> distribution(-1000, 1000);
  MatrixType A = MatrixType::Zero(ak_size, ak_size) ;
  for (int i = 0; i < ak_size; ++i) {
    A(i, i) = distribution(rng);
  }
  return A;
}


template<typename MatrixType>
MatrixType CreateStdRandomSchur(const int ak_size,
                                const int ak_seed = std::time(nullptr)) {
  std::srand(ak_seed);
  MatrixType A = CreateStdRandomTridiagonal<MatrixType>(ak_size, std::rand());
  for (int i = 0; i < ak_size-1; i+= 2) {
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

template<typename VectorType>
VectorType CreateRandomVector(const int ak_size,
    const int ak_seed = std::time(nullptr)) {
  std::srand(ak_seed);
  VectorType v(ak_size);
  v.setRandom();
  return v;
}


// Creating Random Matrices
template<typename MatrixType>
MatrixType CreateUnitaryMatrix(const int ak_size,
    const int ak_seed = std::time(nullptr)) {
  std::srand(ak_seed);
  MatrixType A(ak_size, ak_size);
  Eigen::MatrixXi intMatrix = Eigen::MatrixXi::Random(ak_size, ak_size);
  intMatrix = intMatrix + intMatrix.transpose().eval();
  A = intMatrix.cast<double>();
  Eigen::HouseholderQR<MatrixType> qr(A);
  return qr.householderQ();
}


template<class Matrix>
Matrix RandomSpdMatrix(const int ak_size) {
  //orthogonal matrix
  Matrix A(ak_size, ak_size);
  A.setRandom();
  Eigen::HouseholderQR<Matrix> qr(A);
  // diagonal matrix with positive entries
  Matrix B = Matrix::Identity(ak_size, ak_size);
  for (int i=0; i<ak_size; i++) {
    std::srand(std::time(nullptr)+i);
    double x = std::rand() % 10;
    if (x < 0) B(i,i) = -1/x;
    if (x > 0) B(i,i) = 1/x;
  }
  return qr.householderQ() * B * qr.householderQ().transpose();
}


template<class Matrix>
Matrix RandomNonSymmRealEv(const int ak_size) {
   Matrix B = CreateStdRandom<Matrix>(ak_size);
    B = B.transpose().eval() + B;
    Eigen::MatrixXd C = RandomSpdMatrix<Matrix>(ak_size);
    return B*C;
}


template<class Matrix>
std::enable_if_t<std::is_arithmetic<typename Matrix::Scalar>::value, Matrix>
CreateNormal2x2Matrix(const int ak_seed = std::time(nullptr)) {
  Matrix B(2,2);
  std::mt19937 rng(ak_seed);
  std::uniform_real_distribution<double> distribution(-1000, 1000);
  B(0,0) = distribution(rng);
  B(0,1) = distribution(rng);
  B(1,0) = std::abs(B(0,1));
  B(1,1) = B(0,0);
  return B;
}

template<class Matrix>
std::enable_if_t<!std::is_arithmetic<typename Matrix::Scalar>::value, Matrix>
CreateNormal2x2Matrix(const int ak_seed = std::time(nullptr)) {
  Matrix B(2,2);
  std::mt19937 rng(ak_seed);
  std::uniform_real_distribution<double> distribution(-1000, 1000);
  B(0,0) = std::complex<double>{distribution(rng), distribution(rng)};
  B(0,1) = std::complex<double>{distribution(rng), distribution(rng)};
  if (distribution(rng) > 0) {
    B(1,0) = B(0,1);
  } else {
    B(1,0) = std::conj(B(0,1));
  }
  B(1,1) = B(0,0);
  return B;
}

template<typename MatrixType>
MatrixType CreateStdRandomNormalSchur(const int ak_size,
                                const int ak_seed = std::time(nullptr)) {
  std::srand(ak_seed);
  std::mt19937 rng(ak_seed);
  std::uniform_int_distribution<int> distribution(-100, 100);
  MatrixType A = MatrixType::Zero(ak_size, ak_size);
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnarrowing"
  A(ak_size - 1, ak_size - 1) = typename MatrixType::Scalar{distribution(rng)};
#pragma GCC diagnostic pop
  for (int i = 0; i < ak_size-1; i+= 2) {
    A(Eigen::seqN(i, 2), Eigen::seqN(i, 2)) = CreateNormal2x2Matrix<MatrixType>(std::rand());
    //A(Eigen::seqN(i, 2), Eigen::seqN(i, 2)) = CreateNormal2x2Matrix<MatrixType>(ak_seed);
  }
  return A;
}



template<typename MatrixType, typename T = std::complex<double>>
std::tuple<MatrixType, std::vector<T>>
CreateRandom(const int ak_size, const bool is_symm,
    const int ak_seed = std::time(nullptr)) {
  MatrixType A;
  std::srand(ak_seed);
  std::vector<T> res;
  res.reserve(ak_size);

  if (is_symm) {
    A = CreateStdRandomDiagonal<MatrixType>(ak_size, std::rand());
    std::vector<typename MatrixType::value_type> real_res =
      ConvertToVec(A.diagonal());
    for (auto& x : real_res) {
      res.push_back(T{x});
    }
  } else {
    res.resize(ak_size);
    //A = CreateStdRandomSchur<MatrixType>(ak_size, ak_seed);
    A = CreateStdRandomNormalSchur<MatrixType>(ak_size, std::rand());
    for (int i = 0; i < ak_size-1; ++i) {
      T trace = A(Eigen::seq(i, i+1), Eigen::seq(i, i+1)).trace();
      T tmp = std::sqrt(trace * trace - 4.0 * A(Eigen::seq(i, i+1), Eigen::seq(i, i+1)).determinant());
      T ev1 = (trace + tmp) / 2.0;
      T ev2 = (trace - tmp) / 2.0;
      res.at(i) = ev1;                                                         // First eigenvalue
      ++i;
      res.at(i) = ev2;                                                         // Second eigenvalue
    }
    if (ak_size % 2 == 1) {
      res.at(ak_size - 1) = A(ak_size - 1, ak_size - 1);
    }
  }

  MatrixType Q = CreateUnitaryMatrix<MatrixType>(ak_size, ak_seed);
  A = Q.adjoint() * A * Q;
  return std::tie(A, res);
}

#endif
