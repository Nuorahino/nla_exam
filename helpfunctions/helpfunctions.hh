#ifndef HELPFUNCTIONS_HH
#define HELPFUNCTIONS_HH

#include <type_traits>
#include <complex>


 template <typename T> inline constexpr
 int signum(T x, std::false_type is_signed) {
     return T(0) <= x;
 }

 template <typename T> inline constexpr
 int signum(T x, std::true_type is_signed) {
     return (T(0) <= x) - (x < T(0));
 }

 template <typename T> inline constexpr
 int signum(T x) {
     return signum(x, std::is_signed<T>());
 }

 template <typename T> inline constexpr
 int signum(std::complex<T> x) {
     return signum(x.real(), std::is_signed<T>());
 }

#endif
