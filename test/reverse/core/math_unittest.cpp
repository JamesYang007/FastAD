#define _USE_MATH_DEFINES
#include <type_traits>
#include <fastad_bits/reverse/core/math.hpp>
#include "gtest/gtest.h"

namespace ad {
namespace math {

struct math_fixture : ::testing::Test
{
protected:
    Eigen::VectorXd v;
    Eigen::VectorXd u;

    math_fixture()
        : v(3)
        , u(3)
    {
        v << 1, 0, 0.3;
        u << 1, 2, 3;
    }

    template <class Unary, class F>
    void check_unary_vec(F f)
    {
        Eigen::VectorXd res = Unary::fmap(v.array());
        for (int i = 0; i < v.size(); ++i) {
            EXPECT_DOUBLE_EQ(res(i), f(v(i)));
        }
    }

    template <class Binary, class F>
    void check_binary_vec(F f)
    {
        // test scalar-vector
        Eigen::VectorXd res1 = Binary::fmap(v(0), u.array());
        for (int i = 0; i < v.size(); ++i) {
            EXPECT_DOUBLE_EQ(res1(i), f(v(0), u(i)));
        }

        // test vector-scalar
        Eigen::VectorXd res2 = Binary::fmap(v.array(), u(0));
        for (int i = 0; i < v.size(); ++i) {
            EXPECT_DOUBLE_EQ(res2(i), f(v(i), u(0)));
        }

        // test vector-vector
        Eigen::VectorXd res3 = Binary::fmap(v.array(), u.array());
        for (int i = 0; i < v.size(); ++i) {
            EXPECT_DOUBLE_EQ(res3(i), f(v(i), u(i)));
        }
    }

    template <class ADF, class STDF>
    void test_constant_unary(ADF ad_f, STDF std_f)
    {
        // scalar constant
        auto c = ad_f(ad::constant(1.));
        static_assert(std::is_same_v<
                std::decay_t<decltype(c)>,
                ad::core::Constant<double, ad::scl> >);
        EXPECT_DOUBLE_EQ(c.feval(), std_f(1.));

        // vector constant
        auto vc = ad_f(ad::constant(v));
        static_assert(std::is_same_v<
                std::decay_t<decltype(vc)>,
                ad::core::Constant<double, ad::vec> >);
        auto vc_res = vc.feval();
        for (int i = 0; i < v.size(); ++i) {
            EXPECT_DOUBLE_EQ(vc_res(i), std_f(v(i)));
        }
    }

    template <class ValueType, class ADF, class STDF>
    void test_constant_binary(ADF ad_f, STDF std_f)
    {
        // scalar constant
        auto c = ad_f(ad::constant(1.), ad::constant(2.));
        static_assert(std::is_same_v<
                std::decay_t<decltype(c)>,
                ad::core::Constant<ValueType, ad::scl> >);
        EXPECT_DOUBLE_EQ(c.feval(), std_f(1., 2.));

        // vector constant
        // if checking for boolean operators, use different test vecs
        if constexpr (std::is_same_v<ValueType, bool>) {
            Eigen::Matrix<bool, Eigen::Dynamic, 1> u(3);
            Eigen::Matrix<bool, Eigen::Dynamic, 1> v(3);
            u << true, false, true;
            v << false, true, true;
            auto vc = ad_f(ad::constant(v), ad::constant(u));
            auto vc_res = vc.feval();
            static_assert(std::is_same_v<
                    std::decay_t<decltype(c)>,
                    ad::core::Constant<ValueType, ad::scl> >);
            for (int i = 0; i < v.size(); ++i) {
                EXPECT_DOUBLE_EQ(vc_res(i), std_f(v(i), u(i)));
            }
        } else {
            auto vc = ad_f(ad::constant(v), ad::constant(u));
            auto vc_res = vc.feval();
            static_assert(std::is_same_v<
                    std::decay_t<decltype(c)>,
                    ad::core::Constant<ValueType, ad::scl> >);
            for (int i = 0; i < v.size(); ++i) {
                EXPECT_DOUBLE_EQ(vc_res(i), std_f(v(i), u(i)));
            }
        }
    }
};

////////////////////////////////////////////////////////////
// Unary
////////////////////////////////////////////////////////////

TEST_F(math_fixture, UnaryMinus_scl) 
{
    EXPECT_DOUBLE_EQ(UnaryMinus::fmap(3.), -3.);
    EXPECT_DOUBLE_EQ(UnaryMinus::bmap(3.), -1.);
}

TEST_F(math_fixture, UnaryMinus_vec) 
{
    check_unary_vec<UnaryMinus>([](auto x) { return -x; });
}

TEST_F(math_fixture, Sin_scl) 
{
    EXPECT_DOUBLE_EQ(Sin::fmap(0.), 0.);
    EXPECT_DOUBLE_EQ(Sin::bmap(0.), 1.);
}

TEST_F(math_fixture, Sin_vec) 
{
    check_unary_vec<Sin>([](auto x) { return std::sin(x); });
}

TEST_F(math_fixture, Cos_scl)
{
    EXPECT_DOUBLE_EQ(Cos::fmap(0.), 1.);
    EXPECT_DOUBLE_EQ(Cos::bmap(M_PI/2), -1.);
}

TEST_F(math_fixture, Cos_vec) 
{
    check_unary_vec<Cos>([](auto x) { return std::cos(x); });
}

TEST_F(math_fixture, Tan_scl)
{
    EXPECT_DOUBLE_EQ(Tan::fmap(0.), 0);
    EXPECT_DOUBLE_EQ(Tan::bmap(0.), 1);
}

TEST_F(math_fixture, Tan_vec) 
{
    check_unary_vec<Tan>([](auto x) { return std::tan(x); });
}

TEST_F(math_fixture, Arcsin_scl)
{
    EXPECT_DOUBLE_EQ(Arcsin::fmap(1.), M_PI/2);
    EXPECT_DOUBLE_EQ(Arcsin::bmap(0.), 1.);
}

TEST_F(math_fixture, Arcsin_vec) 
{
    check_unary_vec<Arcsin>([](auto x) { return std::asin(x); });
}

TEST_F(math_fixture, Arccos_scl)
{
    EXPECT_DOUBLE_EQ(Arccos::fmap(1.), 0.);
    EXPECT_DOUBLE_EQ(Arccos::bmap(0.), -1.);
}

TEST_F(math_fixture, Arccos_vec) 
{
    check_unary_vec<Arccos>([](auto x) { return std::acos(x); });
}

TEST_F(math_fixture, Arctan_scl)
{
    EXPECT_DOUBLE_EQ(Arctan::fmap(1.), M_PI/4);
    EXPECT_DOUBLE_EQ(Arctan::bmap(1.), 0.5);
}

TEST_F(math_fixture, Arctan_vec) 
{
    check_unary_vec<Arctan>([](auto x) { return std::atan(x); });
}

TEST_F(math_fixture, Exp_scl)
{
    EXPECT_DOUBLE_EQ(Exp::fmap(0.), 1.);
    EXPECT_DOUBLE_EQ(Exp::bmap(1.), std::exp(1.));
}

TEST_F(math_fixture, Exp_vec) 
{
    check_unary_vec<Exp>([](auto x) { return std::exp(x); });
}

TEST_F(math_fixture, Log_scl)
{
    EXPECT_DOUBLE_EQ(Log::fmap(1.), 0.);
    EXPECT_DOUBLE_EQ(Log::bmap(2.), 0.5);
}

TEST_F(math_fixture, Log_vec) 
{
    check_unary_vec<Log>([](auto x) { return std::log(x); });
}

TEST_F(math_fixture, Id_scl)
{
    EXPECT_DOUBLE_EQ(Id::fmap(1.), 1.);
    EXPECT_DOUBLE_EQ(Id::bmap(2.), 1.);
}

TEST_F(math_fixture, Id_vec) 
{
    check_unary_vec<Id>([](auto x) { return x; });
}

////////////////////////////////////////////////////////////
// Binary
////////////////////////////////////////////////////////////

TEST_F(math_fixture, Add_scl) 
{
    EXPECT_DOUBLE_EQ(Add::fmap(-1.0, 2.1), 1.1);
    EXPECT_DOUBLE_EQ(Add::blmap(-2.01, 2341.2131), 1);
    EXPECT_DOUBLE_EQ(Add::brmap(-2.01, 2341.2131), 1);
}

TEST_F(math_fixture, Add_vec) 
{
    check_binary_vec<Add>([](auto x, auto y) { return x + y; });
}

TEST_F(math_fixture, Sub_scl) 
{
    EXPECT_EQ(Sub::fmap(-1.0, 2.1), -3.1);
    EXPECT_EQ(Sub::blmap(-2.01, 2.), 1.);
    EXPECT_EQ(Sub::brmap(-2.01, 3.), -1.);
}

TEST_F(math_fixture, Sub_vec) 
{
    check_binary_vec<Sub>([](auto x, auto y) { return x - y; });
}

TEST_F(math_fixture, Mul_scl) 
{
    EXPECT_EQ(Mul::fmap(-1.0, 2.1), -2.1);
    EXPECT_EQ(Mul::blmap(-2.01, 2.), 2.);
    EXPECT_EQ(Mul::brmap(-2.01, 3.), -2.01);
}

TEST_F(math_fixture, Mul_vec) 
{
    check_binary_vec<Mul>([](auto x, auto y) { return x * y; });
}

TEST_F(math_fixture, Div_scl)
{
    EXPECT_EQ(Div::fmap(-1.0, 2.1), -1./2.1);
    EXPECT_EQ(Div::blmap(-2.01, 2.), 0.5);
    EXPECT_EQ(Div::brmap(-2.01, 3.), 2.01 / 9.);
}

TEST_F(math_fixture, Div_vec)
{
    check_binary_vec<Div>([](auto x, auto y) { return x / y; });
}

////////////////////////////////////////////////////////////
// Unary Constant Overloads
////////////////////////////////////////////////////////////

TEST_F(math_fixture, constant_operator_unary_minus)
{
    test_constant_unary([](const auto& x) {return -x;}, 
                        [](const auto& x) {return -x;});
}

TEST_F(math_fixture, constant_sin)
{
    test_constant_unary([](const auto& x) {return ad::sin(x);}, 
                        [](const auto& x) {return std::sin(x);});
}

TEST_F(math_fixture, constant_cos)
{
    test_constant_unary([](const auto& x) {return ad::cos(x);}, 
                        [](const auto& x) {return std::cos(x);});
}

TEST_F(math_fixture, constant_tan)
{
    test_constant_unary([](const auto& x) {return ad::tan(x);}, 
                        [](const auto& x) {return std::tan(x);});
}

TEST_F(math_fixture, constant_asin)
{
    test_constant_unary([](const auto& x) {return ad::asin(x);}, 
                        [](const auto& x) {return std::asin(x);});
}

TEST_F(math_fixture, constant_acos)
{
    test_constant_unary([](const auto& x) {return ad::acos(x);}, 
                        [](const auto& x) {return std::acos(x);});
}

TEST_F(math_fixture, constant_atan)
{
    test_constant_unary([](const auto& x) {return ad::atan(x);}, 
                        [](const auto& x) {return std::atan(x);});
}

TEST_F(math_fixture, constant_exp)
{
    test_constant_unary([](const auto& x) {return ad::exp(x);}, 
                        [](const auto& x) {return std::exp(x);});
}

TEST_F(math_fixture, constant_log)
{
    test_constant_unary([](const auto& x) {return ad::log(x);}, 
                        [](const auto& x) {return std::log(x);});
}

TEST_F(math_fixture, constant_id)
{
    test_constant_unary([](const auto& x) {return ad::id(x);}, 
                        [](const auto& x) {return x;});
}

TEST_F(math_fixture, constant_operator_plus)
{
    test_constant_binary<double>([](const auto& x, const auto& y) { return x + y; },
                         [](const auto& x, const auto& y) { return x + y; });
}

TEST_F(math_fixture, constant_operator_minus)
{
    test_constant_binary<double>([](const auto& x, const auto& y) { return x - y; },
                         [](const auto& x, const auto& y) { return x - y; });
}

TEST_F(math_fixture, constant_operator_mult)
{
    test_constant_binary<double>([](const auto& x, const auto& y) { return x * y; },
                         [](const auto& x, const auto& y) { return x * y; });
}

TEST_F(math_fixture, constant_operator_div)
{
    test_constant_binary<double>([](const auto& x, const auto& y) { return x / y; },
                         [](const auto& x, const auto& y) { return x / y; });
}

TEST_F(math_fixture, constant_operator_less)
{
    test_constant_binary<bool>([](const auto& x, const auto& y) { return x < y; },
                         [](const auto& x, const auto& y) { return x < y; });
}

TEST_F(math_fixture, constant_operator_less_eq)
{
    test_constant_binary<bool>([](const auto& x, const auto& y) { return x <= y; },
                         [](const auto& x, const auto& y) { return x <= y; });
}

TEST_F(math_fixture, constant_operator_greater)
{
    test_constant_binary<bool>([](const auto& x, const auto& y) { return x > y; },
                         [](const auto& x, const auto& y) { return x > y; });
}

TEST_F(math_fixture, constant_operator_greater_than)
{
    test_constant_binary<bool>([](const auto& x, const auto& y) { return x >= y; },
                         [](const auto& x, const auto& y) { return x >= y; });
}

TEST_F(math_fixture, constant_operator_eq)
{
    test_constant_binary<bool>([](const auto& x, const auto& y) { return x == y; },
                         [](const auto& x, const auto& y) { return x == y; });
}

TEST_F(math_fixture, constant_operator_neq)
{
    test_constant_binary<bool>([](const auto& x, const auto& y) { return x != y; },
                         [](const auto& x, const auto& y) { return x != y; });
}

TEST_F(math_fixture, constant_operator_and)
{
    test_constant_binary<bool>([](const auto& x, const auto& y) { return LogicalAnd::fmap(x,y); },
                         [](const auto& x, const auto& y) { return LogicalAnd::fmap(x,y); });
}

TEST_F(math_fixture, constant_operator_or)
{
    test_constant_binary<bool>([](const auto& x, const auto& y) { return LogicalOr::fmap(x,y); },
                         [](const auto& x, const auto& y) { return LogicalOr::fmap(x,y); });
}

} // namespace math
} // namespace ad
