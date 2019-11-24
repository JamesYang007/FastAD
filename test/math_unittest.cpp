#define _USE_MATH_DEFINES
#include <fastad_bits/node.hpp>
#include <fastad_bits/math.hpp>
#include "gtest/gtest.h"

namespace ad {
namespace math {

struct admath_fixture : ::testing::Test
{
protected:
};

////////////////////////////////////////////////////////////
// Unary
////////////////////////////////////////////////////////////

TEST_F(admath_fixture, UnaryMinus) 
{
    EXPECT_DOUBLE_EQ(UnaryMinus<double>::fmap(3.), -3.);
    EXPECT_DOUBLE_EQ(UnaryMinus<double>::bmap(3.), -1.);
}

TEST_F(admath_fixture, Sin) 
{
    EXPECT_DOUBLE_EQ(Sin<double>::fmap(0.), 0.);
    EXPECT_DOUBLE_EQ(Sin<double>::bmap(0.), 1.);
}

TEST_F(admath_fixture, Cos)
{
    EXPECT_DOUBLE_EQ(Cos<double>::fmap(0.), 1.);
    EXPECT_DOUBLE_EQ(Cos<double>::bmap(M_PI/2), -1.);
}

TEST_F(admath_fixture, Tan)
{
    EXPECT_DOUBLE_EQ(Tan<double>::fmap(0), 0);
    EXPECT_DOUBLE_EQ(Tan<double>::bmap(0), 1);
}

TEST_F(admath_fixture, Arcsin)
{
    EXPECT_DOUBLE_EQ(Arcsin<double>::fmap(1), M_PI/2);
    EXPECT_DOUBLE_EQ(Arcsin<double>::bmap(0), 1.);
}

TEST_F(admath_fixture, Arccos)
{
    EXPECT_DOUBLE_EQ(Arccos<double>::fmap(1), 0.);
    EXPECT_DOUBLE_EQ(Arccos<double>::bmap(0), -1.);
}

TEST_F(admath_fixture, Arctan)
{
    EXPECT_DOUBLE_EQ(Arctan<double>::fmap(1), M_PI/4);
    EXPECT_DOUBLE_EQ(Arctan<double>::bmap(1), 0.5);
}

TEST_F(admath_fixture, Exp)
{
    EXPECT_DOUBLE_EQ(Exp<double>::fmap(0), 1);
    EXPECT_DOUBLE_EQ(Exp<double>::bmap(1), std::exp(1));
}

TEST_F(admath_fixture, Log)
{
    EXPECT_DOUBLE_EQ(Log<double>::fmap(1), 0);
    EXPECT_DOUBLE_EQ(Log<double>::bmap(2), 0.5);
}

TEST_F(admath_fixture, Id)
{
    EXPECT_DOUBLE_EQ(Id<double>::fmap(1), 1);
    EXPECT_DOUBLE_EQ(Id<double>::bmap(2), 1);
}

////////////////////////////////////////////////////////////
// Binary
////////////////////////////////////////////////////////////

TEST_F(admath_fixture, Add) 
{
    EXPECT_EQ(Add<double>::fmap(-1.0, 2.1), 1.1);
    EXPECT_EQ(Add<double>::blmap(-2.01, 2341.2131), 1);
    EXPECT_EQ(Add<double>::brmap(-2.01, 2341.2131), 1);
}

TEST_F(admath_fixture, Sub) 
{
    EXPECT_EQ(Sub<double>::fmap(-1.0, 2.1), -3.1);
    EXPECT_EQ(Sub<double>::blmap(-2.01, 2.), 1.);
    EXPECT_EQ(Sub<double>::brmap(-2.01, 3.), -1.);
}

TEST_F(admath_fixture, Mul) 
{
    EXPECT_EQ(Mul<double>::fmap(-1.0, 2.1), -2.1);
    EXPECT_EQ(Mul<double>::blmap(-2.01, 2.), 2.);
    EXPECT_EQ(Mul<double>::brmap(-2.01, 3.), -2.01);
}

TEST_F(admath_fixture, Div)
{
    EXPECT_EQ(Div<double>::fmap(-1.0, 2.1), -1./2.1);
    EXPECT_EQ(Div<double>::blmap(-2.01, 2.), 0.5);
    EXPECT_EQ(Div<double>::brmap(-2.01, 3.), 2.01 / 9.);
}

} // namespace math
} // namespace ad
