#ifndef LAPACK_INTERFACE_IMPL_HH
#define LAPACK_INTERFACE_IMPL_HH

#include "lapack_support.hh"

#include <algorithm>
#include <iostream>


template<class Derived, class Mat = Eigen::MatrixXd, class Vec = Eigen::VectorXcd>
std::tuple<Mat, Vec>
CalculateGeneralEigenvalues(const Eigen::MatrixBase<Derived>& ak_matrix, const bool calcEigenvectors = true) {
  typedef double T;
  Mat H = ak_matrix;

  // LAPACK Variables
  char JOBVL = 'N';                   // Dont compute left eigenvalues
  char JOBVR = 'N';                   // Dont compute right eigenvalues
  if (calcEigenvectors) {
    JOBVR = 'V';
  }
  int N = ak_matrix.rows();           // Size of Matrix
  T* A = H.data();               // Matrix
  int LDA = N;                        // is square Matrix
  T* WR = new T[N];         // Real Eigenvalue Part
  T* WL = new T[N];         // Imag Eigenvalue Part
  int LDVL = 1;                       // Size of VL
  T* VL = new T[LDVL * N];  // Only referrenced, when left eigenvector is needed
  int LDVR =  N;                      // Size of the right eigenvector space
  T* VR = new T[LDVR * N];  // Array for storing the right eigenvectors
  int LWORK = 64 * N;                 // Blocksize times N, value currently chosen arbitrarily
  T* WORK = new T[LWORK];   // Workspace
  int INFO;                           // Output 0 on success


  dgeev_(&JOBVL, &JOBVR, &N, A, &LDA, WR, WL, VL, &LDVL, VR, &LDVR, WORK, &LWORK, &INFO);
  //std::cout << "INFO: " << INFO << std::endl;

  Vec eval(N);
  for(int i = 0; i < N; ++i) {
    eval.real()(i) = WR[i];
    eval.imag()(i) = WL[i];
  // Overrite H with the eigenvectors
    for(int ii = 0; ii < N; ++ii) {
      H(i, ii) = VR[i*N + ii];
    }
  }

  delete[] WORK;
  delete[] VR;
  delete[] VL;
  return std::forward_as_tuple(H, eval);
}


template<class Derived, class Mat = Eigen::MatrixXd>
Mat CreateHessenberg(const Eigen::MatrixBase<Derived>& ak_matrix) {
  Mat H = ak_matrix;

  // LAPACK Variables
  int N = ak_matrix.rows();         // Size of Matrix
  int ILO = 1;                      // default
  int IHI = N;                      // default
  double* A = H.data();             // Matrix
  int LDA = N;                      // is square Matrix
  double* TAU = new double[N-1];    // Output scalar factors of elementary reflections
  int LWORK = 64 * N;               // Blocksize times N, value currently chosen arbitrarily
  double* WORK = new double[LWORK]; // Workspace
  int INFO;                         // Output 0 on success


  dgehrd_(&N, &ILO, &IHI, A, &LDA, TAU, WORK, &LWORK, &INFO);
  std::cout << "INFO: " << INFO << std::endl;

  delete[] WORK;
  delete[] TAU;
  return H;
}

template<class Derived, class Mat = Eigen::MatrixXd>
double* CalcEigenvaluesFromHessenberg(const Eigen::MatrixBase<Derived>& ak_matrix,
                                  const double* ak_real_part_evs,
                                  const double* ak_imag_part_evs) {
  Mat mat = ak_matrix;
  int N = ak_matrix.rows();           // Order of the matrix
  char SIDE = 'R';                    // Select the right eigenvector
  char EIGSCR = 'N';                  // No ordering of Evs
  char INITV = 'N';                   // No Initial Vector is provided
  bool* SELECT = new bool[N];         // All Eigenvalues are real or complex conjugates
  std::fill(SELECT, SELECT + N, true);
  double* H = mat.data();             // Hessenberg Matrix
  int LDH = N;                        // Leading Dimension of H, in square case always N
  double* WR = new double[N];         // Real Eigenvalue Part
  std::copy(ak_real_part_evs, ak_real_part_evs + N, WR);
  double* WL = new double[N];         // Imag Eigenvalue Part
  std::copy(ak_imag_part_evs, ak_imag_part_evs + N, WL);
  int LDVL = 1;                       // Size of VL
  double* VL = new double[LDVL * N];  // Only referrenced, when left eigenvector is needed
  int LDVR =  N;                      // Size of the right eigenvector space
  double* VR = new double[LDVR * N];  // Array for storing the right eigenvectors
  int MM = N;                         // Storage space in VR/VL
  int M = N;                          // Columns in VR/VL requirred to store the eigenvectors
  double* WORK = new double[N*(N+2)]; // Work array dim ((N+2)*N)
  int* IFAILL = new int[MM];          // ith entry is 0 if converged successfully
  int* IFAILR = new int[MM];          // ith entry is 0 if converged successfully
  int INFO;                           // 0 on success


  dhsein_(&SIDE, &EIGSCR, &INITV, SELECT, &N, H, &LDH, WR, WL, VL, &LDVL, VR,
      &LDVR, &MM, &M, WORK, IFAILL, IFAILR, &INFO);

  std::cout << INFO << std::endl;
  std::cout << M << std::endl;
  std::cout << "Right" << std::endl;
  std::cout << IFAILR[0] << std::endl;
  std::cout << IFAILR[1] << std::endl;
  std::cout << IFAILR[2] << std::endl;

// TODO find out, why this is an issue
//  delete[] WR;
//  delete[] WL;
//  delete[] VL;
//  delete[] WORK;
//  delete[] IFAILL;
//  delete[] IFAILR;

  return VR;
}



#endif
