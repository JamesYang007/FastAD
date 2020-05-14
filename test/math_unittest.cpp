#define _USE_MATH_DEFINES
#include <type_traits>
#include <fastad_bits/node.hpp>
#include <fastad_bits/math.hpp>
#include "gtest/gtest.h"

namespace ad {
namespace math {

struct admath_fixture : ::testing::Test
{
protected:

    template <class ADF, class STDF>
    void test_constant_unary(ADF ad_f, STDF std_f)
    {
        auto c = ad_f(ad::constant(1.));
        static_assert(std::is_same_v<
                std::decay_t<decltype(c)>,
                ad::core::ConstNode<double> >);
        EXPECT_DOUBLE_EQ(c.feval(), std_f(1.));
        c.beval(1.);
        EXPECT_DOUBLE_EQ(c.get_adjoint(), 0.);
    }

    template <class ADF, class STDF>
    void test_constant_binary(ADF ad_f, STDF std_f)
    {
        auto c = ad_f(ad::constant(1.), ad::constant(2.));
        static_assert(std::is_same_v<
                std::decay_t<decltype(c)>,
                ad::core::ConstNode<double> >);
        EXPECT_DOUBLE_EQ(c.feval(), std_f(1., 2.));
        c.beval(1.);
        EXPECT_DOUBLE_EQ(c.get_adjoint(), 0.);
    }
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

////////////////////////////////////////////////////////////
// Unary Constant Overloads
////////////////////////////////////////////////////////////

TEST_F(admath_fixture, constant_operator_unary_minus)
{
    test_constant_unary([](const auto& x) {return -x;}, 
                        [](const auto& x) {return -x;});
}

TEST_F(admath_fixture, constant_sin)
{
    test_constant_unary([](const auto& x) {return ad::sin(x);}, 
                        [](const auto& x) {return std::sin(x);});
}

TEST_F(admath_fixture, constant_cos)
{
    test_constant_unary([](const auto& x) {return ad::cos(x);}, 
                        [](const auto& x) {return std::cos(x);});
}

TEST_F(admath_fixture, constant_tan)
{
    test_constant_unary([](const auto& x) {return ad::tan(x);}, 
                        [](const auto& x) {return std::tan(x);});
}

TEST_F(admath_fixture, constant_asin)
{
    test_constant_unary([](const auto& x) {return ad::asin(x);}, 
                        [](const auto& x) {return std::asin(x);});
}

TEST_F(admath_fixture, constant_acos)
{
    test_constant_unary([](const auto& x) {return ad::acos(x);}, 
                        [](const auto& x) {return std::acos(x);});
}

TEST_F(admath_fixture, constant_atan)
{
    test_constant_unary([](const auto& x) {return ad::atan(x);}, 
                        [](const auto& x) {return std::atan(x);});
}

TEST_F(admath_fixture, constant_exp)
{
    test_constant_unary([](const auto& x) {return ad::exp(x);}, 
                        [](const auto& x) {return std::exp(x);});
}

TEST_F(admath_fixture, constant_log)
{
    test_constant_unary([](const auto& x) {return ad::log(x);}, 
                        [](const auto& x) {return std::log(x);});
}

TEST_F(admath_fixture, constant_id)
{
    test_constant_unary([](const auto& x) {return ad::id(x);}, 
                        [](const auto& x) {return x;});
}

TEST_F(admath_fixture, constant_operator_plus)
{
    test_constant_binary([](const auto& x, const auto& y) { return x + y; },
                         [](const auto& x, const auto& y) { return x + y; });
}

TEST_F(admath_fixture, constant_operator_minus)
{
    test_constant_binary([](const auto& x, const auto& y) { return x - y; },
                         [](const auto& x, const auto& y) { return x - y; });
}

TEST_F(admath_fixture, constant_operator_mult)
{
    test_constant_binary([](const auto& x, const auto& y) { return x * y; },
                         [](const auto& x, const auto& y) { return x * y; });
}

TEST_F(admath_fixture, constant_operator_div)
{
    test_constant_binary([](const auto& x, const auto& y) { return x / y; },
                         [](const auto& x, const auto& y) { return x / y; });
}

TEST_F(admath_fixture, constant_operator_less)
{
    test_constant_binary([](const auto& x, const auto& y) { return x < y; },
                         [](const auto& x, const auto& y) { return x < y; });
}

TEST_F(admath_fixture, constant_operator_less_eq)
{
    test_constant_binary([](const auto& x, const auto& y) { return x <= y; },
                         [](const auto& x, const auto& y) { return x <= y; });
}

TEST_F(admath_fixture, constant_operator_greater)
{
    test_constant_binary([](const auto& x, const auto& y) { return x > y; },
                         [](const auto& x, const auto& y) { return x > y; });
}

TEST_F(admath_fixture, constant_operator_greater_than)
{
    test_constant_binary([](const auto& x, const auto& y) { return x >= y; },
                         [](const auto& x, const auto& y) { return x >= y; });
}

TEST_F(admath_fixture, constant_operator_eq)
{
    test_constant_binary([](const auto& x, const auto& y) { return x == y; },
                         [](const auto& x, const auto& y) { return x == y; });
}

TEST_F(admath_fixture, constant_operator_neq)
{
    test_constant_binary([](const auto& x, const auto& y) { return x != y; },
                         [](const auto& x, const auto& y) { return x != y; });
}

TEST_F(admath_fixture, constant_operator_and)
{
    test_constant_binary([](const auto& x, const auto& y) { return x && y; },
                         [](const auto& x, const auto& y) { return x && y; });
}

TEST_F(admath_fixture, constant_operator_or)
{
    test_constant_binary([](const auto& x, const auto& y) { return x || y; },
                         [](const auto& x, const auto& y) { return x || y; });
}

} // namespace math
} // namespace ad
