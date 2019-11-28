#include <fastad_bits/dualnum.hpp>
#include "gtest/gtest.h"

namespace ad {
namespace core {

struct dualnum_fixture: ::testing::Test
{
protected:
    DualNum<double> dual;

    dualnum_fixture()
        : dual(2.1, 2.3)
    {}
};

TEST_F(dualnum_fixture, value_type)
{
    bool same_value_type = std::is_same<DualNum<double>::value_type, double>::value;
    EXPECT_TRUE(same_value_type);
}

TEST_F(dualnum_fixture, constructor) 
{
    EXPECT_DOUBLE_EQ(dual.get_value(), 2.1);
    EXPECT_DOUBLE_EQ(dual.get_adjoint(), 2.3);
}

TEST_F(dualnum_fixture, get_set_value) 
{
    dual.set_value(3.4);
    EXPECT_DOUBLE_EQ(dual.get_value(), 3.4);    // value changed
    EXPECT_DOUBLE_EQ(dual.get_adjoint(), 2.3);  // adjoint did not change
}

TEST_F(dualnum_fixture, get_set_adjoint) 
{
    dual.set_adjoint(3.4);
    EXPECT_DOUBLE_EQ(dual.get_value(), 2.1);    // value did not change
    EXPECT_DOUBLE_EQ(dual.get_adjoint(), 3.4);  // adjoint changed
}

} // namespace core
} // namespace ad
