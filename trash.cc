
// Givens rotation for Hessenberg Matrix Needs to be optimized
template<class Matrix, class data_type>
void apply_givens_rotation(Matrix& aA, const int aBegin, const int n, const int j, const int k) {
  data_type r = std::sqrt(std::pow(aA(j, j),2) + std::pow(aA(k, j),2));
  data_type c = aA(j, j)/r;
  data_type s = aA(k,j)/r;
  Matrix Q = Matrix::Identity(n, n); // This is not the best way to implement this
  Q(j - aBegin, j - aBegin) = c;
  Q(k - aBegin, k - aBegin) = c;
  Q(j - aBegin, k - aBegin) = s;
  Q(k - aBegin, j - aBegin) = -s; // complex this is not correct
  //Q(k - aBegin, j - aBegin) = - std::conj(s); // check if correct
  aA(Eigen::seqN(aBegin, n), Eigen::seqN(aBegin, n)) = Q * aA(Eigen::seqN(aBegin, n), Eigen::seqN(aBegin, n)) * Q.transpose();
//  for(int i = aBegin; i < aBegin + n; ++i ) {
//    for(int ii = aBegin; ii < i; ++ii ) {
//      aA(i, ii) = 0;
//    }
//  }
}


  typedef Eigen::SparseMatrix<double> SMat;
  typename Eigen::MatrixXi::Scalar i = 2.3;

  std::cout << std::is_base_of<Eigen::EigenBase<Eigen::MatrixXd>, Eigen::MatrixXd>::value << std::endl;
  std::cout << std::is_same<Eigen::EigenBase<Eigen::MatrixXd>, Eigen::MatrixXd>::value << std::endl;
  std::cout << std::is_base_of<Eigen::SparseMatrixBase<SMat>, SMat>::value << std::endl;
  std::cout << std::is_base_of<Eigen::EigenBase<SMat>, SMat>::value << std::endl;
  std::cout << "last test" << std::endl;
  std::cout << std::is_base_of<Eigen::MatrixBase<SMat>, SMat>::value << std::endl;
  std::cout << typeid(typename Eigen::MatrixXd::Scalar).name() << std::endl;
