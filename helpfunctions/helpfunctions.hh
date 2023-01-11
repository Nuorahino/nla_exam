#ifndef HELPFUNCTIONS_HH
#define HELPFUNCTIONS_HH

#include <type_traits>
#include <complex>
#include <iostream>
#include <vector>


 template <typename T> inline constexpr
 int signum(const T x, std::false_type is_signed) {
     return T(0) <= x;
 }

 template <typename T> inline constexpr
 int signum(const T x, std::true_type is_signed) {
     return (T(0) <= x) - (x < T(0));
 }

 template <typename T> inline constexpr
 int signum(const T x) {
     return signum(x, std::is_signed<T>());
 }

 template <typename T> inline constexpr
 int signum(std::complex<T> x) {
     return signum(x.real(), std::is_signed<T>());
 }


template<class C>
std::ostream & operator<<(std::ostream & out, const std::vector<C> & v)
{
  out << "{";
  size_t last = v.size() - 1;
  for(size_t i = 0; i < v.size(); ++i) {
    out << v[i];
    if (i != last)
      out << ", ";
  }
  out << "}";
  return out;
}

#endif
