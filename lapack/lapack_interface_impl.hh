#ifndef LAPACK_INTERFACE_IMPL_HH
#define LAPACK_INTERFACE_IMPL_HH

#include <algorithm>
#include <iostream>
#include <vector>

#include <eigen3/Eigen/Dense>

#include "lapack_support.hh"
#include "../tests/helpfunctions_for_test.hh"

template<class DataType>
std::vector<std::complex<DataType>>
CalculateTridiagonalEigenvalues(std::vector<DataType> diag, std::vector<DataType> sdiag,
                                const bool calcEigenvectors = false) {
  assert(!calcEigenvectors);

  char COMPZ = 'N';
  int n = diag.size();
  DataType* D = &diag[0];
  DataType* E = &sdiag[0];
  DataType* Z = new DataType[n];
  int LDZ = 1;
  DataType* WORK = new DataType[1];
  int INFO;
  if constexpr (std::is_same<DataType, double>::value) {
    dsteqr_(&COMPZ, &n, D, E, Z, &LDZ, WORK, &INFO);
  } else if constexpr (std::is_same<DataType, float>::value) {
    ssteqr_(&COMPZ, &n, D, E, Z, &LDZ, WORK, &INFO);
  }

  return std::vector<std::complex<DataType>>{D, D + n};
}

template<class DataType, class Derived>
std::tuple<Eigen::Matrix<DataType, -1, -1>, Eigen::Vector<typename ComplexDataType<DataType>::type, -1>>
CalculateGeneralEigenvalues(const Eigen::MatrixBase<Derived>& ak_matrix, const bool calcEigenvectors = false) {
  Eigen::Matrix<DataType, -1, -1> H = ak_matrix;

  // LAPACK Variables
  char JOBVL = 'N';
  char JOBVR = 'N';
  if (calcEigenvectors) {
    JOBVR = 'V';
  }
  int N = H.rows();
  DataType* A = H.data();
  int LDA = N;
  DataType* WR = new DataType[N];
  DataType* WI = new DataType[N];
  int LDVL = 1;
  DataType* VL = new DataType[LDVL * N];
  int LDVR = N;
  DataType* VR = new DataType[LDVR * N];
  int LWORK = 64 * N;
  DataType* WORK = new DataType[LWORK];
  typename RealDataType<DataType>::type* RWORK = new typename RealDataType<DataType>::type[2 * N];
  int INFO;

  if constexpr (std::is_same<DataType, double>::value) {
    dgeev_(&JOBVL, &JOBVR, &N, A, &LDA, WR, WI, VL, &LDVL, VR, &LDVR, WORK, &LWORK, &INFO);
  } else if constexpr (std::is_same<DataType, float>::value) {
    sgeev_(&JOBVL, &JOBVR, &N, A, &LDA, WR, WI, VL, &LDVL, VR, &LDVR, WORK, &LWORK, &INFO);
  } else if constexpr (std::is_same<DataType, std::complex<float>>::value) {
    cgeev_(&JOBVL, &JOBVR, &N, A, &LDA, WR, VL, &LDVL, VR, &LDVR, WORK, &LWORK, RWORK, &INFO);
  } else if constexpr (std::is_same<DataType, std::complex<double>>::value) {
    zgeev_(&JOBVL, &JOBVR, &N, A, &LDA, WR, VL, &LDVL, VR, &LDVR, WORK, &LWORK, RWORK, &INFO);
  }

  Eigen::Vector<typename ComplexDataType<DataType>::type, -1> eval(N);
  if constexpr (IsComplex<DataType>()) {
    for(int i = 0; i < N; ++i) {
      eval(i) = WR[i];
    }
  } else {
    for(int i = 0; i < N; ++i) {
      eval.real()(i) = WR[i];
      eval.imag()(i) = WI[i];
    }
  }
  // Overrite H with the eigenvectors
  for(int i = 0; i < N; ++i) {
    for(int ii = 0; ii < N; ++ii) {
      H(i, ii) = VR[i*N + ii];
    }
  }

  return std::forward_as_tuple(H, eval);
}


template<class DataType, class Derived>
std::tuple<Eigen::Matrix<DataType, -1, -1>, Eigen::Vector<typename ComplexDataType<DataType>::type, -1>>
CalculateHermitianEigenvalues(const Eigen::MatrixBase<Derived>& ak_matrix, const bool calcEigenvectors = false) {
  Eigen::Matrix<DataType, -1, -1> H = ak_matrix;
  typedef typename RealDataType<DataType>::type RDataType;
  typedef typename ComplexDataType<DataType>::type CDataType;

  // LAPACK Variables
  char JOBZ = 'N';
  if (calcEigenvectors) {
    JOBZ = 'V';
  }
  char UPLO = 'U';
  int N = H.rows();
  DataType* A = H.data();
  int LDA = N;
  RDataType* W = new RDataType[N];
  int LWORK = 64 * N;
  DataType* WORK = new DataType[LWORK];
  RDataType* RWORK = new RDataType[3 * N - 2];
  int INFO;

  if constexpr (std::is_same<DataType, double>::value) {
    dsyev_(&JOBZ, &UPLO, &N, A, &LDA, W, WORK, &LWORK, RWORK, &INFO);
  } else if constexpr (std::is_same<DataType, float>::value) {
    //sgeev_(&JOBVL, &JOBVR, &N, A, &LDA, WR, WI, VL, &LDVL, VR, &LDVR, WORK, &LWORK, &INFO);
  } else if constexpr (std::is_same<DataType, std::complex<float>>::value) {
    //cgeev_(&JOBVL, &JOBVR, &N, A, &LDA, WR, VL, &LDVL, VR, &LDVR, WORK, &LWORK, RWORK, &INFO);
  } else if constexpr (std::is_same<DataType, std::complex<double>>::value) {
    zheev_(&JOBZ, &UPLO, &N, A, &LDA, W, WORK, &LWORK, RWORK, &INFO);
  }

  Eigen::Vector<CDataType, -1> eval(N);
  for(int i = 0; i < N; ++i) {
    eval(i) = W[i];
  }
  return std::forward_as_tuple(H, eval);
}
template<bool hermitian, class DataType, class Derived>
std::tuple<Eigen::Matrix<DataType, -1, -1>, Eigen::Vector<typename ComplexDataType<DataType>::type, -1>>
CalculateGeneralEigenvalues(const Eigen::MatrixBase<Derived>& ak_matrix, const bool calcEigenvectors = false) {
  if constexpr (hermitian) {
    return CalculateHermitianEigenvalues<DataType>(ak_matrix, calcEigenvectors);
  } else {
    return CalculateGeneralEigenvalues<DataType>(ak_matrix, calcEigenvectors);
  }
}

template<class DataType, class Derived, class Mat = Eigen::Matrix<DataType, -1, -1>, class RMat = Eigen::Matrix<typename RealDataType<DataType>::type, -1, -1>>
Mat CreateTridiagonal(const Eigen::MatrixBase<Derived>& ak_matrix) {
  typedef typename RealDataType<DataType>::type RDataType;
  Mat M = ak_matrix;

  // LAPACK Variables
  char UPLO = 'L';                  // Storage in Lower part of the matrix
  int N = ak_matrix.rows();         // Size of Matrix
  DataType* A = M.data();             // Matrix
  int LDA = N;                      // is square Matrix
  RDataType* D = new RDataType[N];
  RDataType* E = new RDataType[N-1];
  DataType* TAU = new DataType[N-1];    // Output scalar factors of elementary reflections
  int LWORK = 64 * N;               // Blocksize times N, value currently chosen arbitrarily
  DataType* WORK = new DataType[LWORK]; // Workspace
  int INFO;                         // Output 0 on success


  if constexpr (std::is_same<DataType, double>::value) {
    dsytrd_(&UPLO, &N, A, &LDA, D, E, TAU, WORK, &LWORK, &INFO);
  } else if constexpr (std::is_same<DataType, std::complex<double>>::value) {
    zhetrd_(&UPLO, &N, A, &LDA, D, E, TAU, WORK, &LWORK, &INFO);
  }

  M = Mat::Zero(ak_matrix.rows(), ak_matrix.cols());
  M(0, 0) = D[0];
  for (int i = 1; i < N; ++i) {
    M(i, i) = D[i];
    M(i, i - 1) = E[i-1];
    M(i - 1, i) = E[i-1];
  }

  delete[] WORK;
  delete[] TAU;
  return M;
}

template<class DataType, class Derived, class Mat = Eigen::Matrix<DataType, -1, -1>>
Mat CreateHessenberg(const Eigen::MatrixBase<Derived>& ak_matrix) {
  Mat H = ak_matrix;

  // LAPACK Variables
  int N = ak_matrix.rows();         // Size of Matrix
  int ILO = 1;                      // default
  int IHI = N;                      // default
  DataType* A = H.data();             // Matrix
  int LDA = N;                      // is square Matrix
  DataType* TAU = new DataType[N-1];    // Output scalar factors of elementary reflections
  int LWORK = 64 * N;               // Blocksize times N, value currently chosen arbitrarily
  DataType* WORK = new DataType[LWORK]; // Workspace
  int INFO;                         // Output 0 on success

  if constexpr (std::is_same<DataType, double>::value) {
    dgehrd_(&N, &ILO, &IHI, A, &LDA, TAU, WORK, &LWORK, &INFO);
  } else if constexpr (std::is_same<DataType, std::complex<double>>::value) {
    zgehrd_(&N, &ILO, &IHI, A, &LDA, TAU, WORK, &LWORK, &INFO);
  }

  for (int j = 0; j < N; ++j) {
    for (int i = j + 2; i < N; ++i) {
      H(i, j) = 0;
    }
  }
  delete[] WORK;
  delete[] TAU;
  return H;
}

template<bool Hermitian, class DataType, class Derived, class Mat = Eigen::Matrix<DataType, -1, -1>>
Mat CreateHessenberg(const Eigen::MatrixBase<Derived>& ak_matrix) {
  if constexpr (Hermitian) {
    return CreateTridiagonal<DataType>(ak_matrix);
  }
  return CreateHessenberg<DataType>(ak_matrix);
}

//template<class Derived, class Mat = Eigen::MatrixXd>
//double* CalcEigenvaluesFromHessenberg(const Eigen::MatrixBase<Derived>& ak_matrix,
//                                  const double* ak_real_part_evs,
//                                  const double* ak_imag_part_evs) {
//  Mat mat = ak_matrix;
//  int N = ak_matrix.rows();           // Order of the matrix
//  char SIDE = 'R';                    // Select the right eigenvector
//  char EIGSCR = 'N';                  // No ordering of Evs
//  char INITV = 'N';                   // No Initial Vector is provided
//  bool* SELECT = new bool[N];         // All Eigenvalues are real or complex conjugates
//  std::fill(SELECT, SELECT + N, true);
//  double* H = mat.data();             // Hessenberg Matrix
//  int LDH = N;                        // Leading Dimension of H, in square case always N
//  double* WR = new double[N];         // Real Eigenvalue Part
//  std::copy(ak_real_part_evs, ak_real_part_evs + N, WR);
//  double* WL = new double[N];         // Imag Eigenvalue Part
//  std::copy(ak_imag_part_evs, ak_imag_part_evs + N, WL);
//  int LDVL = 1;                       // Size of VL
//  double* VL = new double[LDVL * N];  // Only referrenced, when left eigenvector is needed
//  int LDVR =  N;                      // Size of the right eigenvector space
//  double* VR = new double[LDVR * N];  // Array for storing the right eigenvectors
//  int MM = N;                         // Storage space in VR/VL
//  int M = N;                          // Columns in VR/VL requirred to store the eigenvectors
//  double* WORK = new double[N*(N+2)]; // Work array dim ((N+2)*N)
//  int* IFAILL = new int[MM];          // ith entry is 0 if converged successfully
//  int* IFAILR = new int[MM];          // ith entry is 0 if converged successfully
//  int INFO;                           // 0 on success
//
//
//  dhsein_(&SIDE, &EIGSCR, &INITV, SELECT, &N, H, &LDH, WR, WL, VL, &LDVL, VR,
//      &LDVR, &MM, &M, WORK, IFAILL, IFAILR, &INFO);
//
//  std::cout << INFO << std::endl;
//  std::cout << M << std::endl;
//  std::cout << "Right" << std::endl;
//  std::cout << IFAILR[0] << std::endl;
//  std::cout << IFAILR[1] << std::endl;
//  std::cout << IFAILR[2] << std::endl;
//
//// TODO find out, why this is an issue
////  delete[] WR;
////  delete[] WL;
////  delete[] VL;
////  delete[] WORK;
////  delete[] IFAILL;
////  delete[] IFAILR;
//
//  return VR;
//}

template<class DataType>
std::vector<DataType> compute_givens_parameter(DataType a, DataType b) {
  typename RealDataType<DataType>::type c;
  DataType s, r;
  if constexpr(std::is_same<DataType, double>::value) {
    dlartg_(&a,&b,&c,&s,&r);
  } else if constexpr (std::is_same<DataType, float>::value) {
    slartg_(&a,&b,&c,&s,&r);
  } else if constexpr (std::is_same<DataType, std::complex<float>>::value) {
    clartg_(&a,&b,&c,&s,&r);
  } else if constexpr (std::is_same<DataType, std::complex<double>>::value) {
    zlartg_(&a,&b,&c,&s,&r);
  }
  return std::vector<DataType>{c,s,r};
}

template<class DataType>
void compute_givens_parameter(DataType a, DataType b, std::array<DataType, 3> &entries) {
  typename RealDataType<DataType>::type c;
  if constexpr(std::is_same<DataType, double>::value) {
    dlartg_(&a,&b,&c,&entries.at(1),&entries.at(2));
  } else if constexpr (std::is_same<DataType, float>::value) {
    slartg_(&a,&b,&c,&entries.at(1),&entries.at(2));
  } else if constexpr (std::is_same<DataType, std::complex<float>>::value) {
    clartg_(&a,&b,&c,&entries.at(1),&entries.at(2));
  } else if constexpr (std::is_same<DataType, std::complex<double>>::value) {
    zlartg_(&a,&b,&c,&entries.at(1),&entries.at(2));
  }
  entries.at(0) = c;
  return;
}


//template<class DataType, class Derived>
//Eigen::Matrix<DataType, -1, -1> apply_givens_right(const Eigen::MatrixBase<Derived>& ak_matrix, int k, int j,
//    typename RealDataType<DataType>::type c, DataType s) {
//  Eigen::Matrix<DataType, -1, -1> H = ak_matrix;
//  s = complex_conj(s);          // As the function for applying the rotation does not have a left, and right side
//
//  int n = ak_matrix.rows();
//  Eigen::Matrix<DataType, -1, 1> X_vec = H(Eigen::all, k);
//  Eigen::Matrix<DataType, -1, 1> Y_vec = H(Eigen::all, j);
//  DataType* X = X_vec.data();
//  int INCX = 1;
//  DataType* Y = Y_vec.data();
//  int INCY = 1;
//  int INCC = 1;
//  typename RealDataType<DataType>::type* C_vec = new typename RealDataType<DataType>::type[n];
//  DataType* S_vec = new DataType[n];
//  for(int i = 0; i < n; ++i) {
//    S_vec[i] = s;
//    C_vec[i] = c;
//  }
//
//  if constexpr(std::is_same<DataType, double>::value) {
//    dlartv_(&n, X, &INCX, Y, &INCY, C_vec, S_vec, &INCC);
//  } else if constexpr (std::is_same<DataType, float>::value) {
//    slartv_(&n, X, &INCX, Y, &INCY, C_vec, S_vec, &INCC);
//  } else if constexpr (std::is_same<DataType, std::complex<float>>::value) {
//    clartv_(&n, X, &INCX, Y, &INCY, C_vec, S_vec, &INCC);
//  } else if constexpr (std::is_same<DataType, std::complex<double>>::value) {
//    zlartv_(&n, X, &INCX, Y, &INCY, C_vec, S_vec, &INCC);
//  }
//
//  H(Eigen::all, k) = X_vec;
//  H(Eigen::all, j) = Y_vec;
//  return H;
//}
//
//template<class DataType, class Derived>
//Eigen::Matrix<DataType, -1, -1> apply_givens_right(const Eigen::MatrixBase<Derived>& ak_matrix, int k,
//    typename RealDataType<DataType>::type c, DataType s) {
//  return apply_givens_right(ak_matrix, k, k + 1, c, s);
//}
//
//template<class DataType, class Derived>
//Eigen::Matrix<DataType, -1, -1> apply_givens_left(const Eigen::MatrixBase<Derived>& ak_matrix, int k, int j,
//                                                  typename RealDataType<DataType>::type c, DataType s) {
//  Eigen::Matrix<DataType, -1, -1> H = ak_matrix;
//
//  int n = ak_matrix.cols();
//  Eigen::Matrix<DataType, -1, 1> X_vec = H(k, Eigen::all);
//  Eigen::Matrix<DataType, -1, 1> Y_vec = H(j, Eigen::all);
//  DataType* X = X_vec.data();
//  int INCX = 1;
//  DataType* Y = Y_vec.data();
//  int INCY = 1;
//  int INCC = 1;
//  typename RealDataType<DataType>::type* C_vec = new typename RealDataType<DataType>::type[n];
//  DataType* S_vec = new DataType[n];
//  for (int i = 0; i < n; ++i) {
//    S_vec[i] = s;
//    C_vec[i] = c;
//  }
//
//  if constexpr (std::is_same<DataType, double>::value) {
//    dlartv_(&n, X, &INCX, Y, &INCY, C_vec, S_vec, &INCC);
//  } else if constexpr (std::is_same<DataType, float>::value) {
//    slartv_(&n, X, &INCX, Y, &INCY, C_vec, S_vec, &INCC);
//  } else if constexpr (std::is_same<DataType, std::complex<float>>::value) {
//    clartv_(&n, X, &INCX, Y, &INCY, C_vec, S_vec, &INCC);
//  } else if constexpr (std::is_same<DataType, std::complex<double>>::value) {
//    zlartv_(&n, X, &INCX, Y, &INCY, C_vec, S_vec, &INCC);
//  }
//
//  H(k, Eigen::all) = X_vec;
//  H(j, Eigen::all) = Y_vec;
//  return H;
//}
//
//template <class DataType, class Derived>
//Eigen::Matrix<DataType, -1, -1>
//apply_givens_left(const Eigen::MatrixBase<Derived>& ak_matrix, int k,
//                  typename RealDataType<DataType>::type c, DataType s) {
//  return apply_givens_left(ak_matrix, k, k + 1, c, s);
//}
//
template <class DataType, class Derived>
std::tuple<Eigen::Matrix<DataType, -1, -1>, DataType>
get_householder(const Eigen::MatrixBase<Derived>& ak_v) {
  int n = ak_v.rows();
  DataType alpha = ak_v(0);
  Eigen::Vector<DataType, -1> v = ak_v(Eigen::seqN(1, n-1));
  DataType* vd = v.data();
  int incrx = 1;
  DataType tau;
  if constexpr (std::is_same<DataType, double>::value) {
    dlarfg_(&n, &alpha, vd, &incrx, &tau);
  } else if constexpr (std::is_same<DataType, float>::value) {
    slarfg_(&n, &alpha, vd, &incrx, &tau);
  } else if constexpr (std::is_same<DataType, std::complex<float>>::value) {
    clarfg_(&n, &alpha, vd, &incrx, &tau);
  } else if constexpr (std::is_same<DataType, std::complex<double>>::value) {
    zlarfg_(&n, &alpha, vd, &incrx, &tau);
  }
  return {v, tau};
}

//template <class DataType, class Derived, class Derived2>
//Eigen::Matrix<DataType, -1, -1>
//apply_householder_left(const Eigen::MatrixBase<Derived>& ak_matrix, const Eigen::MatrixBase<Derived2>& ak_w, DataType tau) {
//  Eigen::Matrix<DataType, -1, -1> res = ak_matrix;
//  Eigen::Vector<DataType, -1> w = ak_w;
//  char side = 'L';
//  int m = ak_matrix.rows();
//  int n = ak_matrix.cols();
//  DataType* V = w.data();
//  int INCV = 1;
//
//  DataType* C = res.data();
//  int LDC = m;
//  DataType* WORK = new DataType[n];
//
//  if constexpr (std::is_same<DataType, double>::value) {
//    dlarf_(&side, &m, &n, V, &INCV, &tau, C, &LDC, WORK);
//  } else if constexpr (std::is_same<DataType, float>::value) {
//    slarf_(&side, &m, &n, V, &INCV, &tau, C, &LDC, WORK);
//  } else if constexpr (std::is_same<DataType, std::complex<float>>::value) {
//    clarf_(&side, &m, &n, V, &INCV, &tau, C, &LDC, WORK);
//  } else if constexpr (std::is_same<DataType, std::complex<double>>::value) {
//    zlarf_(&side, &m, &n, V, &INCV, &tau, C, &LDC, WORK);
//  }
//  return res;
//}
//
//template <class DataType, class Derived, class Derived2>
//Eigen::Matrix<DataType, -1, -1>
//apply_householder_right(const Eigen::MatrixBase<Derived>& ak_matrix, const Eigen::MatrixBase<Derived2>& ak_w, DataType tau) {
//  Eigen::Matrix<DataType, -1, -1> res = ak_matrix;
//  Eigen::Vector<DataType, -1> w = ak_w;
//  char side = 'R';
//  int m = ak_matrix.rows();
//  int n = ak_matrix.cols();
//  DataType* V = w.data();
//  int INCV = 1;
//
//  DataType* C = res.data();
//  int LDC = m;
//  DataType* WORK = new DataType[n];
//
//  if constexpr (std::is_same<DataType, double>::value) {
//    dlarf_(&side, &m, &n, V, &INCV, &tau, C, &LDC, WORK);
//  } else if constexpr (std::is_same<DataType, float>::value) {
//    slarf_(&side, &m, &n, V, &INCV, &tau, C, &LDC, WORK);
//  } else if constexpr (std::is_same<DataType, std::complex<float>>::value) {
//    clarf_(&side, &m, &n, V, &INCV, &tau, C, &LDC, WORK);
//  } else if constexpr (std::is_same<DataType, std::complex<double>>::value) {
//    zlarf_(&side, &m, &n, V, &INCV, &tau, C, &LDC, WORK);
//  }
//  return res;
//}

#endif
