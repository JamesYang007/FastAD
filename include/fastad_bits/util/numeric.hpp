#pragma once
#include <limits>

namespace ad {
namespace util{

template <class T>
inline constexpr T inf = 
    std::numeric_limits<T>::is_iec559 ? 
    std::numeric_limits<T>::infinity() :
    std::numeric_limits<T>::max();

template <class T>
inline constexpr T neg_inf = 
    std::numeric_limits<T>::is_iec559 ? 
    -std::numeric_limits<T>::infinity() :
    std::numeric_limits<T>::lowest();

} // namespace util
} // namespace ad
