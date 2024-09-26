#ifndef SFINAE_HH_
#define SFINAE_HH_

#include <type_traits>

typedef char SFINAE_yes;
struct SFINAE_no {char _[2];};

template <typename T>
class HasElementType {
  template <typename U, typename>
    class HelperClass{};

  template <typename X>
    static SFINAE_yes Test(HelperClass<X,typename X::ElementType>*);

  template <typename X>
    static SFINAE_no Test(...);

  public:
  enum { value = sizeof(Test<T>(0)) == sizeof(SFINAE_yes)};
};

template <typename T>
class HasScalar {
  template <typename U, typename>
    class HelperClass{};

  template <typename X>
    static SFINAE_yes Test(HelperClass<X,typename X::Scalar>*);

  template <typename X>
    static SFINAE_no Test(...);

  public:
  enum { value = sizeof(Test<T>(0)) == sizeof(SFINAE_yes)};
};

template <typename T>
class Haselem_type {
  template <typename U, typename>
    class HelperClass{};

  template <typename X>
    static SFINAE_yes Test(HelperClass<X,typename X::elem_type>*);

  template <typename X>
    static SFINAE_no Test(...);

  public:
  enum { value = sizeof(Test<T>(0)) == sizeof(SFINAE_yes)};
};



template <typename, typename = void>
struct ElementType {};

template <typename T>
struct ElementType<T, std::enable_if_t<HasElementType<T>::value, void>> {
  using type = typename T::ElementType;
};

template <typename T>
struct ElementType<T, std::enable_if_t<!HasElementType<T>::value && HasScalar<T>::value, void>> {
  using type = typename T::Scalar;
};

template <typename T>
struct ElementType<T, std::enable_if_t<!HasElementType<T>::value && !HasScalar<T>::value && Haselem_type<T>::value, void>> {
  using type = typename T::elem_type;
};

template <typename T>
class HasRowsFunction {
  template <typename U, std::size_t (U::*)() const>
    class HelperClass{};

  template <typename X>
    static SFINAE_yes Test(HelperClass<X, &X::rows>*);

  template <typename X>
    static SFINAE_no Test(...);

  public:
  enum { value = sizeof(Test<T>(0)) == sizeof(SFINAE_yes)};
};

template <typename T>
class Hasn_rows {
  template <typename U, typename>
    class HelperClass{};

  template <typename X>
    static SFINAE_yes Test(HelperClass<X, std::enable_if_t<std::is_same<decltype(std::declval<X>().n_rows), std::size_t>::value|| std::is_same<decltype(std::declval<X>().n_rows), const unsigned long long>::value, void>>*);

  template <typename X>
    static SFINAE_no Test(...);

  public:
  enum { value = sizeof(Test<T>(0)) == sizeof(SFINAE_yes)};
};

template <typename T>
std::enable_if_t<HasRowsFunction<T>::value, decltype(std::declval<T>().rows())>
  inline static rows(const T &Mat) {
    return Mat.rows();
}

template <typename T>
std::enable_if_t<!HasRowsFunction<T>::value && Hasn_rows<T>::value, decltype(std::declval<T>().n_rows)>
  inline static rows(const T &Mat) {
    return Mat.n_rows;
};


#endif
