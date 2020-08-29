#pragma once
#include <Eigen/Dense>

namespace ad {
namespace util {

// Size pack is just an alias for Eigen array of size_t.
// Stack-allocated 2x1 (column) vector.
using SizePack = Eigen::Array<size_t, 2, 1>;

} // namespace util
} // namespace ad
