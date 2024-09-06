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

template< typename... Ts >
using Void_t = void;

template< typename, typename = void > struct ElementType;

template< typename T, typename >
struct ElementType
{
   using Type = T;
};

template< typename T >
struct ElementType< T, Void_t< typename T::ElementType > >
{
   using Type = typename T::ElementType;
};

template< typename T >
struct ElementType< T, Void_t< typename T::Scalar > >
{
   using Type = typename T::Scalar;
};
#endif
