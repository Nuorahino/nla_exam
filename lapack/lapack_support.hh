#ifndef LAPACK_SUPPORT_HH
#define LAPACK_SUPPORT_HH
// dgeev_ is a symbol in the LAPACK library files
extern "C" {
extern int dgeev_(char*,
                  char*,
                  int*,
                  double*,
                  int*,
                  double*,
                  double*,
                  double*,
                  int*,
                  double*,
                  int*,
                  double*,
                  int*,
                  int*);
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

#endif
