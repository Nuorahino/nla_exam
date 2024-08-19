#ifndef HELPFUNCTIONS_FOR_TEST_HH
#define HELPFUNCTIONS_FOR_TEST_HH

#include <iostream>
#include <type_traits>
#include <catch2/catch_all.hpp>
#include <complex>

#include "../include/helpfunctions.hh"
#include "test_hungarian.hh"

template<class DataType>
struct RealDataType {
};

template<>
struct RealDataType<double> {
  using type = double;
};

template<>
struct RealDataType<float> {
  using type = float;
};

template<>
struct RealDataType<std::complex<double>> {
  using type = double;
};

template<>
struct RealDataType<std::complex<float>> {
  using type = float;
};

template<class DataType>
struct ComplexDataType {
};

template<>
struct ComplexDataType<double> {
  using type = std::complex<double>;
};

template<>
struct ComplexDataType<float> {
  using type = std::complex<float>;
};

template<>
struct ComplexDataType<std::complex<double>> {
  using type = std::complex<double>;
};

template<>
struct ComplexDataType<std::complex<float>> {
  using type = std::complex<float>;
};

template<class DataType>
std::enable_if_t<!IsComplex<DataType>(), DataType>
get_real(DataType x) {
  return x;
}
template<class DataType>
std::enable_if_t<IsComplex<DataType>(), typename RealDataType<DataType>::type>
get_real(DataType x) {
  return x.real();
}

template<class DataType>
DataType complex_conj(DataType x) {
  if constexpr (IsComplex<DataType>()) {
    return std::conj(x);
  }
  return x;
}


template<class DataType>
double tol() {
  if constexpr(std::is_same<DataType, double>::value) {
    return 1e-12;
  } else if constexpr (std::is_same<DataType, float>::value) {
    return 1e-6;
  } else if constexpr (std::is_same<DataType, std::complex<float>>::value) {
    return 1e-6;
  } else if constexpr (std::is_same<DataType, std::complex<double>>::value) {
    return 1e-12;
  }
  return 0;
}

template<class DataType, int n>
DataType inline gen_random() {
  if constexpr(std::is_same<DataType, double>::value) {
    return GENERATE(take(n, random(-100, 100)));
  } else if constexpr (std::is_same<DataType, float>::value) {
    return GENERATE(take(n, random(-100, 100)));
  } else if constexpr (std::is_same<DataType, std::complex<float>>::value) {
    return DataType{GENERATE(take(n, random(-100, 100))), GENERATE(take(n, random(-100, 100)))};
  } else if constexpr (std::is_same<DataType, std::complex<double>>::value) {
    return DataType{GENERATE(take(n, random(-100, 100))), GENERATE(take(n, random(-100, 100)))};
  }
  return GENERATE(take(n, random(-100, 100)));
}

#define TEST  \
int n = 10; \
int min = 0;  \
int max = 8; \
TestType new_gen_random() { \
  if constexpr(std::is_same<TestType, int>::value) { \
    return GENERATE(take(n, random(min, max))); \
  } else if constexpr(std::is_same<TestType, double>::value) { \
    return GENERATE(take(n, random(min, max))); \
  } else if constexpr (std::is_same<TestType, float>::value) { \
    return GENERATE(take(n, random(min, max))); \
  } else if constexpr (std::is_same<TestType, std::complex<float>>::value) { \
    return TestType{GENERATE(take(n, random(min, max))), GENERATE(take(n, random(min, max)))}; \
  } else if constexpr (std::is_same<DataType, std::complex<double>>::value) { \
    return TestType{GENERATE(take(n, random(min, max))), GENERATE(take(n, random(min, max)))}; \
  } \
  return GENERATE(take(n, random(-100, 100))); \
}

template <typename T>
class ComplexGenerator : public Catch::Generators::IGenerator<T> {
    T current_number;
    std::minstd_rand m_rand;
    std::uniform_real_distribution<typename RealDataType<T>::type> dist;

public:
    ComplexGenerator(double min, double max)
      : m_rand(std::random_device{}()),
        dist(min, max)
    {
      static_cast<void>(next());
    }

    bool next() override
    {
      if constexpr(IsComplex<T>()) {
        current_number = T{dist(m_rand), dist(m_rand)};
      } else {
        current_number = dist(m_rand);
      }
        return true;
    }

    T const& get() const override
    {
        return current_number;
    }
};

template <typename T>
Catch::Generators::GeneratorWrapper<T> ComplexRandom(double min, double max)
{
      return Catch::Generators::GeneratorWrapper<T>(Catch::Detail::make_unique<ComplexGenerator<T>>(min, max));
}


template <class DataType>
void min_matching(std::vector<DataType>& a, const std::vector<DataType>& b) {
  assert(a.size() == b.size());

  while ( true ) {
    double min = std::numeric_limits<double>::max();
    // Initialize Graph network
    std::vector<std::vector<double>> graph(b.size());
    for (unsigned int i = 0; i < a.size(); ++i) {
      for (unsigned int ii = 0; ii < b.size(); ++ii) {
        double x = std::abs(b.at(ii) - a.at(i)) - std::abs(b.at(i) - a.at(i));
        if ( x < min ) min = x;
        graph.at(i).push_back(x);
      }
    }
    // Ensure that the graph is non negative to apply the Hungarian method
    for (unsigned int i = 0; i < a.size(); ++i) {
      for (unsigned int ii = 0; ii < b.size(); ++ii) {
        graph.at(i).at(ii) -= min;
      }
    }
    std::vector<int> hungry_res = hungarian(graph);
    std::vector<DataType> tmp = a;
    bool optimal = true;
    for(int i = 0; i < static_cast<int>(a.size()); ++i) {
      if ( hungry_res.at(i) != i) {
        a.at(i) = tmp.at(hungry_res.at(i));
        optimal = false;
      }
    }
    if (optimal) break;
  }
}

#endif
