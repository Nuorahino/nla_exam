#ifndef SFINAE_HH_
#define SFINAE_HH_

#include <type_traits>

//template< typename... Ts >
//using Void_t = void;
//
//template< typename, typename = void > struct UnderlyingElementHelper;
//
//template< typename T, typename >
//struct UnderlyingElementHelper
//{
//   using Type = T;
//};
//
//template< typename T >
//struct UnderlyingElementHelper< T, Void_t< typename T::ElementType > >
//{
//   using Type = typename T::ElementType;
//};
//
//template< typename T >
//struct UnderlyingElementHelper< T, Void_t< typename T::Scalar > >
//{
//   using Type = typename T::Scalar;
//};

template <typename T>
using SFINAE_helper = void;

typedef char SFINAE_yes;
struct SFINAE_no {char _[2];};

template <typename, typename = void>
struct ElementType;

template <typename T>
struct ElementType<T, SFINAE_helper<typename T::ElementType> > {
  using Type = typename T::ElementType;
};

template <typename T>
struct ElementType<T, SFINAE_helper<std::enable_if_t<
                          !std::is_same<SFINAE_helper<typename T::ElementType>, void>(),
                          typename T::Scalar> > > {
  using Type = typename T::Scalar;
};

template <typename T>
class HasRowsFunction {
  template <typename U, std::size_t (U::*)() const>
    class HelperClass{};

  template <typename X>
    static char Test(HelperClass<X, &X::rows> *);

  template <typename X>
    static SFINAE_no Test(...);

  public:
  enum { value = sizeof(Test<T>(0)) == sizeof(SFINAE_yes)};
};

template <typename, typename = void>
struct Hasn_rows {
  bool value = false;
};

template <typename T>
struct Hasn_rows<T, SFINAE_helper<typename T::n_rows> > {
  bool value = true;
};


template <typename T>
std::enable_if_t<HasRowsFunction<T>::value ,std::size_t>
  rows(T &Mat) {
    return Mat.rows();
}

template <typename T>
std::enable_if_t<!HasRowsFunction<T>::value && Hasn_rows<T>::value ,std::size_t>
  rows(T &Mat) {
    return Mat.n_rows;
};

#endif
