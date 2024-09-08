#ifndef EIGEN_WRAPPER_
#define EIGEN_WRAPPER_

template <class Matrix>
class EigenWrapper {
  public:
    typedef typename Matrix::Scalar Scalar;

    explicit EigenWrapper(const Matrix &A) : Mat(A) {};

    std::size_t rows() const { return Mat.rows(); }

    int trace() const { return Mat.trace(); }

    int determinant() const { return Mat.determinant(); }

    Scalar& operator()(const unsigned i, const unsigned j) { return Mat(i, j); }

    Scalar operator()(const unsigned i, const unsigned j) const { return Mat(i, j); }


    Matrix Mat;
};
#endif
