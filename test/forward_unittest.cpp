#define _USE_MATH_DEFINES
#include <fastad_bits/forward.hpp>
#include "gtest/gtest.h"

namespace ad {

struct adforward_fixture: ::testing::Test
{
protected:
};

////////////////////////////////////////////////////////////
// Unary
////////////////////////////////////////////////////////////

TEST_F(adforward_fixture, negate) 
{
    ForwardVar<double> x(2);
    x.set_adjoint(1);   // set direction
    EXPECT_DOUBLE_EQ(-x.get_value(), -2.);
    EXPECT_DOUBLE_EQ(-x.get_adjoint(), -1.);
}

TEST_F(adforward_fixture, sin) 
{
    ForwardVar<double> x(0);
    x.set_adjoint(1);   // set direction
    ForwardVar<double> res = ad::sin(x);
    EXPECT_DOUBLE_EQ(res.get_value(), 0.);
    EXPECT_DOUBLE_EQ(res.get_adjoint(), 1.);
}

TEST_F(adforward_fixture, cos)
{
    ForwardVar<double> x(0);
    x.set_adjoint(1);   // set direction
    ForwardVar<double> res = ad::cos(x);
    EXPECT_DOUBLE_EQ(res.get_value(), 1.);
    EXPECT_DOUBLE_EQ(res.get_adjoint(), 0.);
}

TEST_F(adforward_fixture, tan)
{
    ForwardVar<double> x(0);
    x.set_adjoint(1);   // set direction
    ForwardVar<double> res = ad::tan(x);
    EXPECT_DOUBLE_EQ(res.get_value(), 0.);
    EXPECT_DOUBLE_EQ(res.get_adjoint(), 1.);
}

TEST_F(adforward_fixture, asin)
{
    ForwardVar<double> x(0);
    x.set_adjoint(1);   // set direction
    ForwardVar<double> res = ad::asin(x);
    EXPECT_DOUBLE_EQ(res.get_value(), 0);
    EXPECT_DOUBLE_EQ(res.get_adjoint(), 1.);
}

TEST_F(adforward_fixture, acos)
{
    ForwardVar<double> x(0);
    x.set_adjoint(1);   // set direction
    ForwardVar<double> res = ad::acos(x);
    EXPECT_DOUBLE_EQ(res.get_value(), M_PI/2);
    EXPECT_DOUBLE_EQ(res.get_adjoint(), -1.);
}

TEST_F(adforward_fixture, atan)
{
    ForwardVar<double> x(1);
    x.set_adjoint(1);   // set direction
    ForwardVar<double> res = ad::atan(x);
    EXPECT_DOUBLE_EQ(res.get_value(), M_PI/4);
    EXPECT_DOUBLE_EQ(res.get_adjoint(), 0.5);
}

TEST_F(adforward_fixture, exp)
{
    ForwardVar<double> x(0);
    x.set_adjoint(1);   // set direction
    ForwardVar<double> res = ad::exp(x);
    EXPECT_DOUBLE_EQ(res.get_value(), 1);
    EXPECT_DOUBLE_EQ(res.get_adjoint(), 1);
}

TEST_F(adforward_fixture, log)
{
    ForwardVar<double> x(2);
    x.set_adjoint(1);   // set direction
    ForwardVar<double> res = ad::log(x);
    EXPECT_DOUBLE_EQ(res.get_value(), std::log(2));
    EXPECT_DOUBLE_EQ(res.get_adjoint(), 0.5);
}

TEST_F(adforward_fixture, sqrt)
{
    ForwardVar<double> x(4);
    x.set_adjoint(1);   // set direction
    ForwardVar<double> res = ad::sqrt(x);
    EXPECT_DOUBLE_EQ(res.get_value(), 2.);
    EXPECT_DOUBLE_EQ(res.get_adjoint(), 0.25);
}

////////////////////////////////////////////////////////////
// Binary
////////////////////////////////////////////////////////////

TEST_F(adforward_fixture, add) 
{
    ForwardVar<double> x(4, 1), y(3, 1);
    ForwardVar<double> res = x + y;
    EXPECT_DOUBLE_EQ(res.get_value(), 7);
    EXPECT_DOUBLE_EQ(res.get_adjoint(), 2);    // directional derivative in direction (1,1)
}

TEST_F(adforward_fixture, sub) 
{
    ForwardVar<double> x(4, 1), y(3, 1);
    ForwardVar<double> res = x - y;
    EXPECT_DOUBLE_EQ(res.get_value(), 1);
    EXPECT_DOUBLE_EQ(res.get_adjoint(), 0);    // directional derivative in direction (1,1)
}

TEST_F(adforward_fixture, mul) 
{
    ForwardVar<double> x(4, 1), y(3, -1);
    ForwardVar<double> res = x * y;
    EXPECT_DOUBLE_EQ(res.get_value(), 12);
    EXPECT_DOUBLE_EQ(res.get_adjoint(), -1);    // directional derivative in direction (1,-1)
}

TEST_F(adforward_fixture, div)
{
    ForwardVar<double> x(4, 1), y(3, -1);
    ForwardVar<double> res = x / y;
    EXPECT_DOUBLE_EQ(res.get_value(), 4./3);
    EXPECT_DOUBLE_EQ(res.get_adjoint(), 1./3 + 4./9);    // directional derivative in direction (1,1)
}

} // namespace ad
