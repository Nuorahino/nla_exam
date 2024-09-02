#ifndef LAPACK_SUPPORT_HH
#define LAPACK_SUPPORT_HH

#include <complex.h>

extern "C" {
extern int dsteqr_(char*,               // Compz: "N" no eigenvalues
                   int*,                // N: order of the Matrix
                   double*,             // D: double precision array of diagonal entries
                   double*,             // E: double precision array of subdiagonal entries
                   double*,             // Z: double preciosion array when no eigenvalues are computed of size n
                   int*,                // LDZ: without eigenvalues 1
                   double*,             // WORK: double precision array with Compz = N unreferenzed
                   int*);               // INFO: Output


extern int sgeev_(char*,                // jobvl: 'V' left eigenvectors are computed, 'V' not
                  char*,                // jobvr: 'V' right eigenvectors are computed, 'V' not
                  int*,                 // N: order of the matrix
                  float*,               // A: array of dim (LDA, N), at the end overwritten
                  int*,                 // LDA: Leading dimension of A >= max{1, N}
                  float*,               // WR: Array of dim N, output Real part of computed eigenvalues
                  float*,               // WI: Array of dim N, output Real part of computed eigenvalues
                  float*,               // vl: Array of dim (LDVL, N), if JOBVL = 'V' ourput left eigenvectors
                  int*,                 // LDVL: >= 1, if JOBVL, then >= N
                  float*,               // vr: Array of dim (LDVR, N), if JOBVR = 'V' ourput right eigenvectors
                  int*,                 // LDVR: >= 1, if JOBVR, then >= N
                  float*,               // WORK: Array of dim (Max{1, LWORK}), if INFO = 0, WORK(1) is optimal LWORK
                  int*,                 // LWORK: Dimension of WORk array, >= 4N generally chosen larger for performance
                  int*);                // INFO: 0(success), -i < 0 (i-th argmuent has illegal value), i > 0(computed till i)
}

extern "C" {
extern int dgeev_(char*,                // jobvl: 'V' left eigenvectors are computed, 'V' not
                  char*,                // jobvr: 'V' right eigenvectors are computed, 'V' not
                  int*,                 // N: order of the matrix
                  double*,              // A: array of dim (LDA, N), at the end overwritten
                  int*,                 // LDA: Leading dimension of A >= max{1, N}
                  double*,              // WR: Array of dim N, output Real part of computed eigenvalues
                  double*,              // WI: Array of dim N, output Real part of computed eigenvalues
                  double*,              // vl: Array of dim (LDVL, N), if JOBVL = 'V' ourput left eigenvectors
                  int*,                 // LDVL: >= 1, if JOBVL, then >= N
                  double*,              // vr: Array of dim (LDVR, N), if JOBVR = 'V' ourput right eigenvectors
                  int*,                 // LDVR: >= 1, if JOBVR, then >= N
                  double*,              // WORK: Array of dim (Max{1, LWORK}), if INFO = 0, WORK(1) is optimal LWORK
                  int*,                 // LWORK: Dimension of WORk array, >= 4N generally chosen larger for performance
                  int*);                // INFO: 0(success), -i < 0 (i-th argmuent has illegal value), i > 0(computed till i)
}

extern "C" {
extern int cgeev_(char*,                // jobvl: 'V' left eigenvectors are computed, 'V' not
                  char*,                // jobvr: 'V' right eigenvectors are computed, 'V' not
                  int*,                 // N: order of the matrix
                  std::complex<float>*, // A: array of dim (LDA, N), at the end overwritten
                  int*,                 // LDA: Leading dimension of A >= max{1, N}
                  std::complex<float>*, // W: Array of dim N, output computed eigenvalues
                  std::complex<float>*, // vl: Array of dim (LDVL, N), if JOBVL = 'V' ourput left eigenvectors
                  int*,                 // LDVL: >= 1, if JOBVL, then >= N
                  std::complex<float>*, // vr: Array of dim (LDVR, N), if JOBVR = 'V' ourput right eigenvectors
                  int*,                 // LDVR: >= 1, if JOBVR, then >= N
                  std::complex<float>*, // WORK: Array of dim (Max{1, LWORK}), if INFO = 0, WORK(1) is optimal LWORK
                  int*,                 // LWORK: Dimension of WORk array, >= 4N generally chosen larger for performance
                  float*,               // RWORK: Array of dim (2 * N)
                  int*);                // INFO: 0(success), -i < 0 (i-th argmuent has illegal value), i > 0(computed till i)
}

extern "C" {
extern int zgeev_(char*,                // jobvl: 'V' left eigenvectors are computed, 'V' not
                  char*,                // jobvr: 'V' right eigenvectors are computed, 'V' not
                  int*,                 // N: order of the matrix
                  std::complex<double>*,// A: array of dim (LDA, N), at the end overwritten
                  int*,                 // LDA: Leading dimension of A >= max{1, N}
                  std::complex<double>*,// W: Array of dim N, output computed eigenvalues
                  std::complex<double>*,// vl: Array of dim (LDVL, N), if JOBVL = 'V' ourput left eigenvectors
                  int*,                 // LDVL: >= 1, if JOBVL, then >= N
                  std::complex<double>*,// vr: Array of dim (LDVR, N), if JOBVR = 'V' ourput right eigenvectors
                  int*,                 // LDVR: >= 1, if JOBVR, then >= N
                  std::complex<double>*,// WORK: Array of dim (Max{1, LWORK}), if INFO = 0, WORK(1) is optimal LWORK
                  int*,                 // LWORK: Dimension of WORk array, >= 4N generally chosen larger for performance
                  double*,              // RWORK: Array of dim (2 * N)
                  int*);                // INFO: 0(success), -i < 0 (i-th argmuent has illegal value), i > 0(computed till i)
}


// Symmetric Eigenvalue solver
extern "C" {
extern int dgees_(char*,
                  char*,
                  int*,
                  int*,
                  double*,
                  int*,
                  int*,
                  double*,
                  double*,
                  double*,
                  int*,
                  double*,
                  int*,
                  bool*,
                  int*); // SELECT missing after the first two chars
}

// Compute the Eigenvectors of a real symmetric tridiagonal matrix T corresponding to specified eigenvalues, using inverse iteration
extern "C" {
extern int sstein_(int*,    // N: INT order of Matrix
                  double*,  // D: Real n diagonal elements
                  double*,  // E: Real n-1 subdiagonal elements
                  int*,     // M: Number of eigenvectors to be found
                  double*,  // W: Ordered Eigenvalues like order = 'B'
                  int*,     // IBLOCK:
                  int*,     // ISPLIT:
                  double*,  // Z:
                  int*,     // LDZ:
                  double*,  // WORK:
                  int*,     // IWORK:
                  int*,     // IFAIL:
                  int*      // INFO:
                  );
}

// Compute the Eigenvectors of a real Hessenberg Matrix
extern "C" {
extern int shsein_(char*,    // SIDE: which evecs to compute ('R' for right)
                  char*,    // EIGSRC: Ordering of Evs ('N' for no ordering)
                  char*,    // INITV: 'N' no initial vector, 'U' user supplied inital vector
                  int*,     // N: The order of the Matrix
                  double*,  // H: The upper Hessenberg Matrix
                  int*,     // LDH: Leading diminson (N if symmetric)
                  double*,  // WR: Real Evs (dim N)
                  double*,  // WL: Imag Evs (dim N)
                  double*,  // VL:
                  int*,     // LDVL: leading dimension of the array VL
                  double*,  // VR:
                  int*,     // LDVR: N if SIDE = R or B
                  int*,     // MM: columns in VL or VR
                  int*,     // M: Numer of columns in VL or VR to store Evecs
                  double*,  // WORK: Array dim (N+2)*N
                  int*,     // IFAILL: dim MM, not referenced for SIDE 'R'
                  int*,     // IFAILR: dim MM, IFAILR(i) = j > 0, if the ith eigenvector failed, 0 else
                  int*      // INFO: 0 if succes < 0 ~ -i if i is illegal, > 0 = i if number of evec failed to converge
                  );
}

extern "C" {
extern int dhsein_(char*,    // SIDE: which evecs to compute ('R' for right)
                  char*,    // EIGSRC: Ordering of Evs ('N' for no ordering)
                  char*,    // INITV: 'N' no initial vector, 'U' user supplied inital vector
                  bool*,    // SELECT: dimension n, TRUE if EV and Evec real, also for complex conjugate evs
                  int*,     // N: The order of the Matrix
                  double*,  // H: The upper Hessenberg Matrix
                  int*,     // LDH: Leading diminson (N if symmetric)
                  double*,  // WR: Real Evs (dim N)
                  double*,  // WL: Imag Evs (dim N)
                  double*,  // VL:
                  int*,     // LDVL: leading dimension of the array VL
                  double*,  // VR:
                  int*,     // LDVR: N if SIDE = R or B
                  int*,     // MM: columns in VL or VR
                  int*,     // M: Numer of columns in VL or VR to store Evecs
                  double*,  // WORK: Array dim (N+2)*N
                  int*,     // IFAILL: dim MM, not referenced for SIDE 'R'
                  int*,     // IFAILR: dim MM, IFAILR(i) = j > 0, if the ith eigenvector failed, 0 else
                  int*      // INFO: 0 if succes < 0 ~ -i if i is illegal, > 0 = i if number of evec failed to converge
                  );
}

// reduce general matrix to upper Hessenberg
extern "C" {
extern int sgehrd_(
    int*,           // N: Order of the Matrix
    int*,           // ILO: default 1
    int*,           // IHI: default N
    double*,        // A: On entry Matrix, on exit upper Triangular Matrix below Hessenberg representation of Q with tau
    int*,           // LDA: Leading Dimension of A (N)
    double*,        // TAU: Scalar Factors of elementary reflectors (dim N-1)
    double*,        // WORK: dim (LWORK) WORK(1) returns optimal LWORK
    int*,           // LWORK: N * NB, NB is Blocksize
    int*            // INFO: 0 on Success, < 0 = -i, if ith argument is illegal
    );
}

// reduce general matrix to upper Hessenberg
extern "C" {
extern int dgehrd_(
    int*,           // N: Order of the Matrix
    int*,           // ILO: default 1
    int*,           // IHI: default N
    double*,        // A: On entry Matrix, on exit upper Triangular Matrix below Hessenberg representation of Q with tau
    int*,           // LDA: Leading Dimension of A (N)
    double*,        // TAU: Scalar Factors of elementary reflectors (dim N-1)
    double*,        // WORK: dim (LWORK) WORK(1) returns optimal LWORK
    int*,           // LWORK: N * NB, NB is Blocksize
    int*            // INFO: 0 on Success, < 0 = -i, if ith argument is illegal
    );
}

// Generate the Givens Rotation in real single precision
extern "C" {
extern int slartg_(
  float*,             // First entry
  float*,             // Second entry
  float*,             // C entry
  float*,             // S entry
  float*             // R entry
  );
}

// Generate the Givens Rotation in real double precision
extern "C" {
extern int dlartg_(
  double*,             // First entry
  double*,             // Second entry
  double*,             // C entry
  double*,             // S entry
  double*             // R entry
  );
}

extern "C" {
extern int clartg_(
  std::complex<float>*,             // First entry
  std::complex<float>*,             // Second entry
  float *,                     // C entry
  std::complex<float>*,             // S entry
  std::complex<float>*              // R entry
  );
}

// Generate the Givens Rotation in complex double precision
extern "C" {
extern int zlartg_(
  std::complex<double>*,             // First entry
  std::complex<double>*,             // Second entry
  double*,                      // C entry
  std::complex<double>*,             // S entry
  std::complex<double>*              // R entry
  );
}

extern "C" {
extern int dlasr_(
  char*,                        // Side: 'L' or 'R'
  char*,                        // Pivot: 'V' rows(k, k+1), 'T' rows (1, k) or 'B'  rows (k,z)
  char*,                        // Direct: 'F' Forwards (apply first rotation first) or 'B'
  int*,                         // M: Number of rows > 1
  int*,                         // N: Number of cols > 1
  double*,                      // C: Cosine of the rotation of Dim (M-1) if 'L' or (N-1) for 'R'
  double*,                      // S: Sine of the rotation of Dim (M-1) if 'L' or (N-1) for 'R'
  double*,                      // A: MAtrix of size (LDA, N), Overwritten in the output
  int*                          // LDA: Leading Dimenstion of the Matrix >= max{1,m}
  );
}

extern "C" {
extern int slasr_(
  char*,                        // Side: 'L' or 'R'
  char*,                        // Pivot: 'V' rows(k, k+1), 'T' rows (1, k) or 'B'  rows (k,z)
  char*,                        // Direct: 'F' Forwards (apply first rotation first) or 'B'
  int*,                         // M: Number of rows > 1
  int*,                         // N: Number of cols > 1
  float*,                       // C: Cosine of the rotation of Dim (M-1) if 'L' or (N-1) for 'R'
  float*,                       // S: Sine of the rotation of Dim (M-1) if 'L' or (N-1) for 'R'
  float*,                       // A: MAtrix of size (LDA, N), Overwritten in the output
  int*                          // LDA: Leading Dimenstion of the Matrix >= max{1,m}
  );
}

extern "C" {
extern int clasr_(
  char*,                        // Side: 'L' or 'R'
  char*,                        // Pivot: 'V' rows(k, k+1), 'T' rows (1, k) or 'B'  rows (k,z)
  char*,                        // Direct: 'F' Forwards (apply first rotation first) or 'B'
  int*,                         // M: Number of rows > 1
  int*,                         // N: Number of cols > 1
  float*,                       // C: Cosine of the rotation of Dim (M-1) if 'L' or (N-1) for 'R'
  float*,                       // S: Sine of the rotation of Dim (M-1) if 'L' or (N-1) for 'R'
  std::complex<float>*,         // A: MAtrix of size (LDA, N), Overwritten in the output
  int*                          // LDA: Leading Dimenstion of the Matrix >= max{1,m}
  );
}

extern "C" {
extern int zlasr_(
  char*,                        // Side: 'L' or 'R'
  char*,                        // Pivot: 'V' rows(k, k+1), 'T' rows (1, k) or 'B'  rows (k,z)
  char*,                        // Direct: 'F' Forwards (apply first rotation first) or 'B'
  int*,                         // M: Number of rows > 1
  int*,                         // N: Number of cols > 1
  double*,                      // C: Cosine of the rotation of Dim (M-1) if 'L' or (N-1) for 'R'
  double*,                      // S: Sine of the rotation of Dim (M-1) if 'L' or (N-1) for 'R'
  std::complex<double>*,        // A: MAtrix of size (LDA, N), Overwritten in the output
  int*                          // LDA: Leading Dimenstion of the Matrix >= max{1,m}
  );
}

extern "C" {
extern int clartv_(
  int*,                         // n: Number of plane rotations to apply
  std::complex<float>*,         // X: Complex Array of Dim (1 + (N-1) * INCX)
  int*,                         // INCX: Increment of X (default is 1)
  std::complex<float>*,         // Y: Complex Array of Dim (1 + (N-1) * INCY)
  int*,                         // INCY: Increment of Y (default is 1)
  float*,                      // C: Cosine of the rotation of Dim (M-1) if 'L' or (N-1) for 'R'
  std::complex<float>*,        // S: Sine of the rotation of Dim (M-1) if 'L' or (N-1) for 'R'
  int*                          // INCC: Increment between elements in C and S
  );
}

extern "C" {
extern int zlartv_(
  int*,                         // n: Number of plane rotations to apply
  std::complex<double>*,         // X: Complex Array of Dim (1 + (N-1) * INCX)
  int*,                         // INCX: Increment of X (default is 1)
  std::complex<double>*,         // Y: Complex Array of Dim (1 + (N-1) * INCY)
  int*,                         // INCY: Increment of Y (default is 1)
  double*,                      // C: Cosine of the rotation of Dim (M-1) if 'L' or (N-1) for 'R'
  std::complex<double>*,        // S: Sine of the rotation of Dim (M-1) if 'L' or (N-1) for 'R'
  int*                          // INCC: Increment between elements in C and S
  );
}

extern "C" {
extern int slartv_(
  int*,                         // n: Number of plane rotations to apply
  float*,         // X: Complex Array of Dim (1 + (N-1) * INCX)
  int*,                         // INCX: Increment of X (default is 1)
  float*,         // Y: Complex Array of Dim (1 + (N-1) * INCY)
  int*,                         // INCY: Increment of Y (default is 1)
  float*,                      // C: Cosine of the rotation of Dim (M-1) if 'L' or (N-1) for 'R'
  float*,        // S: Sine of the rotation of Dim (M-1) if 'L' or (N-1) for 'R'
  int*                          // INCC: Increment between elements in C and S
  );
}

extern "C" {
extern int dlartv_(
  int*,                         // n: Number of plane rotations to apply
  double*,         // X: Complex Array of Dim (1 + (N-1) * INCX)
  int*,                         // INCX: Increment of X (default is 1)
  double*,         // Y: Complex Array of Dim (1 + (N-1) * INCY)
  int*,                         // INCY: Increment of Y (default is 1)
  double*,                      // C: Cosine of the rotation of Dim (M-1) if 'L' or (N-1) for 'R'
  double*,        // S: Sine of the rotation of Dim (M-1) if 'L' or (N-1) for 'R'
  int*                          // INCC: Increment between elements in C and S
  );
}


// TODO add applying Givens rotation
// lasr: series of plane rotations to a general rectangular matrix
// largv: vector of plane rotations
// lartv: apply vector of plane rotations to vectors
// lar2v apply vector of plane rotations to 2x2 matrix from both sides to symmetric/Hermitian matrix (Interesting!)


// Generate the Householder Reflection in double precision
extern "C" {
extern int dlarfg_(             // add a (p) add the end, to make beta positive
  int*,                         // size n
  double*,                      // first entry (alpha), Output beta
  double*,                      // rest of the vector (size n-1), on output last n-1 entries of  Householder
  int*,                         // increment of x ????
  double*                       // Tau, H = I - tau (1, v) * (1, v)^T
  );
}

extern "C" {
extern int slarfg_(             // add a (p) add the end, to make beta positive
  int*,                         // size n
  float*,                      // first entry (alpha), Output beta
  float*,                      // rest of the vector (size n-1), on output last n-1 entries of  Householder
  int*,                         // increment of x ????
  float*                       // Tau, H = I - tau (1, v) * (1, v)^T
  );
}

extern "C" {
extern int zlarfg_(             // add a (p) add the end, to make beta positive
  int*,                         // size n
  std::complex<double>*,                      // first entry (alpha), Output beta
  std::complex<double>*,                      // rest of the vector (size n-1), on output last n-1 entries of  Householder
  int*,                         // increment of x ????
  std::complex<double>*                       // Tau, H = I - tau (1, v) * (1, v)^T
  );
}

extern "C" {
extern int clarfg_(             // add a (p) add the end, to make beta positive
  int*,                         // size n
  std::complex<float>*,                      // first entry (alpha), Output beta
  std::complex<float>*,                      // rest of the vector (size n-1), on output last n-1 entries of  Householder
  int*,                         // increment of x ????
  std::complex<float>*                       // Tau, H = I - tau (1, v) * (1, v)^T
  );
}

extern "C" {
extern int dlarf_(
  char*,                        // Side: 'L' or 'R'
  int*,                         // M: Number of rows > 1
  int*,                         // N: Number of cols > 1
  double*,                      // v: representation of H array 1 + (M-1) abc(INCV) if SIDE = 'L', else N instead of M
  int*,                         // incv: Increment between elements of v <> 0
  double*,                      // tau: same in the mathematic representation
  double*,                      // c: array with dim (LDC, N), On entry C, on exit result
  int*,                         // ldc: Leading dimension of the array C (LDC >= max(1, M))
  double*                       // work: Leading Dimenstion of the Matrix >= max{1,m}
  );
}

extern "C" {
extern int slarf_(
  char*,                        // Side: 'L' or 'R'
  int*,                         // M: Number of rows > 1
  int*,                         // N: Number of cols > 1
  float*,                      // v: representation of H array 1 + (M-1) abc(INCV) if SIDE = 'L', else N instead of M
  int*,                         // incv: Increment between elements of v <> 0
  float*,                      // tau: same in the mathematic representation
  float*,                      // c: array with dim (LDC, N), On entry C, on exit result
  int*,                         // ldc: Leading dimension of the array C (LDC >= max(1, M))
  float*                       // work: Leading Dimenstion of the Matrix >= max{1,m}
  );
}

extern "C" {
extern int zlarf_(
  char*,                        // Side: 'L' or 'R'
  int*,                         // M: Number of rows > 1
  int*,                         // N: Number of cols > 1
  std::complex<double>*,        // v: representation of H array 1 + (M-1) abc(INCV) if SIDE = 'L', else N instead of M
  int*,                         // incv: Increment between elements of v <> 0
  std::complex<double>*,                      // tau: same in the mathematic representation
  std::complex<double>*,                      // c: array with dim (LDC, N), On entry C, on exit result
  int*,                         // ldc: Leading dimension of the array C (LDC >= max(1, M))
  std::complex<double>*                       // work: Leading Dimenstion of the Matrix >= max{1,m}
  );
}

extern "C" {
extern int clarf_(
  char*,                        // Side: 'L' or 'R'
  int*,                         // M: Number of rows > 1
  int*,                         // N: Number of cols > 1
  std::complex<float>*,         // v: representation of H array 1 + (M-1) abc(INCV) if SIDE = 'L', else N instead of M
  int*,                         // incv: Increment between elements of v <> 0
  std::complex<float>*,                      // tau: same in the mathematic representation
  std::complex<float>*,                      // c: array with dim (LDC, N), On entry C, on exit result
  int*,                         // ldc: Leading dimension of the array C (LDC >= max(1, M))
  std::complex<float>*                       // work: Leading Dimenstion of the Matrix >= max{1,m}
  );
}

#endif
