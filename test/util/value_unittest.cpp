#include <gtest/gtest.h>
#include <fastad_bits/util/value.hpp>

namespace ad {
namespace util {

struct value_fixture : ::testing::Test
{
protected:
};

TEST_F(value_fixture, ones_scl)
{
    double x = 0.;
    ones(x);
    EXPECT_DOUBLE_EQ(x, 1.);
}

} // namespace util
} // namespace ad
