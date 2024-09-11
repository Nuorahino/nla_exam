#ifndef HERM_QR_HH_
#define HERM_QR_HH_

#include <type_traits>

#include "helpfunctions.hh"



template <class DataType, bool is_hermitian, class Matrix>
std::enable_if_t<!std::is_arithmetic<typename ElementType<Matrix>::type>::value || !is_hermitian, std::vector<DataType>>
QrIterationHessenberg(const Matrix &a_matrix, const double ak_tol = 1e-12) {

#endif
