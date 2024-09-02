#ifndef MATRXI_CLASSES_HH
#define MATRXI_CLASSES_HH

#include <vector>
#include <cassert>

#include <eigen3/Eigen/Dense>

template<class DT>
class tridiagonal_matrix2{
  private:
    std::vector<DT> diag;
    std::vector<DT> sdiag;

  public:
    typedef DT Scalar;

    tridiagonal_matrix2() = default;

    explicit tridiagonal_matrix2(const Eigen::Matrix<DT, -1, -1> &A) {
      diag.resize(A.rows());
      sdiag.resize(A.rows());
      for(int i = 0; i < A.rows() - 1; ++i) {
        diag.at(i) = A(i,i);
        sdiag.at(i) = A(i+1,i);
      }
        diag.at(A.rows() - 1) = A(A.rows() - 1,A.rows() - 1);
    }

    ~tridiagonal_matrix2() = default;


    DT &operator() (const unsigned i, const unsigned j) {
      if (i == j) {
        return diag.at(i);
      } else if ( i == j + 1) {
        return sdiag.at(j);
      } else if ( i == j - 1) {
        return sdiag.at(i);
      }
      assert(false);
    }


    const DT &operator() (const unsigned i, const unsigned j) const {
      if (i == j) {
        return diag.at(i);
      } else if ( i == j + 1) {
        return sdiag.at(j);
      } else if ( i == j - 1) {
        return sdiag.at(i);
      }
      assert(false);
    }

    unsigned size() const {
      return diag.size();
    }
};


template<class DT>
class tridiagonal_matrix{
  private:
    std::vector<std::array<DT, 2>> data;

  public:
    typedef DT Scalar;

    tridiagonal_matrix() = default;

    explicit tridiagonal_matrix(const Eigen::Matrix<DT, -1, -1> &A) {
      data.resize(A.rows());
      for(int i = 0; i < A.rows() - 1; ++i) {
        //data.push_back({A(i,i), A(i + 1, i)});
        data.at(i)[0] = A(i,i);
        data.at(i)[1] = A(i+1,i);
      }
      unsigned i = A.rows() - 1;
      data.at(i)[0] = A(i,i);
        //data.push_back({A(A.rows() - 1,A.rows() - 1), 0});
    }

    ~tridiagonal_matrix() = default;

    DT& operator() (const unsigned i, const unsigned j) {
      if (i == j) {
        return data.at(i)[0];
      } else if ( i == j + 1) {
        return data.at(j)[1];
      } else if ( i == j - 1) {
        return data.at(i)[1];
      }
      assert(false);
    }

    const DT& operator() (const unsigned i, const unsigned j) const {
      if (i == j) {
        return data.at(i)[0];
      } else if ( i == j + 1) {
        return data.at(j)[1];
      } else if ( i == j - 1) {
        return data.at(i)[1];
      }
      assert(false);
    }

    unsigned size() const {
      return data.size();
    }
};


#endif
