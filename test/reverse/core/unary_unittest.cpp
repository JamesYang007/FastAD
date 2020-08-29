#define _USE_MATH_DEFINES
#include "gtest/gtest.h"
#include <array>
#include <fastad_bits/reverse/core/unary.hpp>
#include <testutil/base_fixture.hpp>

namespace ad {
namespace core {

struct unary_fixture : base_fixture
{
protected:
    using unary_t = MockUnary;
    using scl_unary_t = UnaryNode<unary_t, scl_expr_view_t>;
    using vec_unary_t = UnaryNode<unary_t, vec_expr_view_t>;
    using mat_unary_t = UnaryNode<unary_t, mat_expr_view_t>;
    using scl_scl_unary_t = UnaryNode<unary_t, scl_unary_t>;

    scl_unary_t scl_unary;
    vec_unary_t vec_unary;
    mat_unary_t mat_unary;
    scl_scl_unary_t scl_scl_unary;

    aVectorXd vseed;
    value_t seed = 3.14;

    aVectorXd v;
    aVectorXd u;
    Eigen::VectorXd val_buf;
    Eigen::VectorXd adj_buf;

    unary_fixture()
        : base_fixture()
        , scl_unary(scl_expr)
        , vec_unary(vec_expr)
        , mat_unary(mat_expr)
        , scl_scl_unary(scl_unary)
        , vseed(3)
        , v(3)
        , u(3)
        , val_buf()
        , adj_buf()
    {
        vseed << 2.13, 0.3231, 4.231;
        v << 1, 0, 0.3;
        u << 1, 2, 3;

        // IMPORTANT: bind value for unary nodes.
        // Heuristic: vec and mat have largest requirement on cache, 
        // so take max between the two and should suffice for all other exprs.
        auto size_pack = vec_unary.bind_cache_size();
        size_pack = size_pack.max(mat_unary.bind_cache_size());
        val_buf.resize(size_pack(0));
        adj_buf.resize(size_pack(1));
        ptr_pack_t ptr_pack(val_buf.data(), adj_buf.data());
        scl_unary.bind_cache(ptr_pack);
        vec_unary.bind_cache(ptr_pack);
        mat_unary.bind_cache(ptr_pack);
        scl_scl_unary.bind_cache(ptr_pack);
    }

    template <class Unary, class S, class T>
    void test_unary_vec(const S& seed,
                        const Eigen::ArrayBase<T>& val,
                        const Eigen::ArrayBase<T>& f,
                        const Eigen::ArrayBase<T>& adj) {
        aVectorXd res = Unary::fmap(val);
        check_eq(res, f);
        res.setZero();
        res.array() += Unary::bmap(seed, val, f);
        check_eq(res, adj);
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

};

////////////////////////////////////////////////////////////////////////
// UnaryNode TEST
////////////////////////////////////////////////////////////////////////

TEST_F(unary_fixture, scl_feval) 
{
    check_eq(scl_unary.feval(), unary_t::fmap(scl_expr.get()));
}

TEST_F(unary_fixture, scl_beval) 
{
    scl_unary.beval(seed); 
    check_eq(scl_expr.get_adj(), unary_t::bmap(seed,0,0)); 
}

TEST_F(unary_fixture, vec_feval) 
{
    auto& res = vec_unary.feval();
    aVectorXd actual = aVectorXd::NullaryExpr(
            vec_size, [&](size_t i) { return unary_t::fmap(vec_expr.get()(i)); });
    check_eq(res, actual);
}

TEST_F(unary_fixture, vec_beval) 
{
    aVectorXd vseed(vec_unary.size());
    vseed.setZero();
    vseed(1) = seed;
    vseed(3) = seed;
    vec_unary.beval(vseed);
    aVectorXd actual = aVectorXd::NullaryExpr(vec_size, 
            [&](size_t i) { 
                return (i == 1 || i == 3) ? unary_t::bmap(seed,0,0) : 0.; 
            });
    check_eq(vec_expr.get_adj(), actual); 
}

TEST_F(unary_fixture, mat_feval) 
{
    auto& res = mat_unary.feval();
    Eigen::MatrixXd actual = Eigen::MatrixXd::NullaryExpr(mat_rows, mat_cols, 
            [&](size_t i, size_t j) { 
                return unary_t::fmap(mat_expr.get()(i,j)); 
            });
    check_eq(res, actual);
}

TEST_F(unary_fixture, mat_beval) 
{
    mat_unary.beval(seed);
    Eigen::MatrixXd actual(mat_rows, mat_cols);
    actual.array() = unary_t::bmap(seed,0,0);
    check_eq(mat_expr.get_adj(), actual); 
}

TEST_F(unary_fixture, scl_scl_feval) 
{
    check_eq(scl_scl_unary.feval(), 
             unary_t::fmap(unary_t::fmap(scl_expr.get())));
}

TEST_F(unary_fixture, scl_scl_beval) 
{
    scl_scl_unary.beval(seed);
    check_eq(scl_expr.get_adj(0,0), 
             unary_t::bmap(unary_t::bmap(seed,0,0),0,0)); 
}

////////////////////////////////////////////////////////////////////////
// Struct TEST
////////////////////////////////////////////////////////////////////////

TEST_F(unary_fixture, UnaryMinus_scl) 
{
    check_eq(UnaryMinus::fmap(3.), -3.);
    check_eq(UnaryMinus::bmap(seed,0,-3.), -seed);
}

TEST_F(unary_fixture, UnaryMinus_vec) 
{
    aVectorXd f = -v;
    aVectorXd adj = -seed * aVectorXd::Ones(v.size());
    aVectorXd vadj = -vseed;
    test_unary_vec<UnaryMinus>(seed, v, f, adj);
    test_unary_vec<UnaryMinus>(vseed, v, f, vadj);
}

TEST_F(unary_fixture, Sin_scl) 
{
    check_eq(Sin::fmap(0.), 0.);
    check_eq(Sin::bmap(seed,0.,0.), seed);
}

TEST_F(unary_fixture, Sin_vec) 
{
    aVectorXd f = v.sin();
    aVectorXd adj = seed * v.cos();
    aVectorXd vadj = vseed * v.cos();
    test_unary_vec<Sin>(seed, v, f, adj);
    test_unary_vec<Sin>(vseed, v, f, vadj);
}

TEST_F(unary_fixture, Cos_scl)
{
    check_eq(Cos::fmap(0.), 1.);
    check_eq(Cos::bmap(seed,M_PI/2,0.), -seed);
}

TEST_F(unary_fixture, Cos_vec) 
{
    aVectorXd f = v.cos();
    aVectorXd adj = -seed * v.sin();
    aVectorXd vadj = -vseed * v.sin();
    test_unary_vec<Cos>(seed, v, f, adj);
    test_unary_vec<Cos>(vseed, v, f, vadj);
}

TEST_F(unary_fixture, Tan_scl)
{
    check_eq(Tan::fmap(0.), 0.);
    check_eq(Tan::bmap(seed,0.,0.), seed);
}

TEST_F(unary_fixture, Tan_vec) 
{
    aVectorXd f = v.tan();
    aVectorXd adj = seed / (v.cos() * v.cos());
    aVectorXd vadj = vseed / (v.cos() * v.cos());
    test_unary_vec<Tan>(seed, v, f, adj);
    test_unary_vec<Tan>(vseed, v, f, vadj);
}

TEST_F(unary_fixture, Arcsin_scl)
{
    check_eq(Arcsin::fmap(1.), M_PI/2);
    check_eq(Arcsin::bmap(seed, 0., Arcsin::fmap(0.)), seed);
}

TEST_F(unary_fixture, Arcsin_vec) 
{
    aVectorXd f = v.asin();
    aVectorXd adj = seed / (1. - v * v).sqrt();
    aVectorXd vadj = vseed / (1. - v * v).sqrt();
    test_unary_vec<Arcsin>(seed, v, f, adj);
    test_unary_vec<Arcsin>(vseed, v, f, vadj);
}

TEST_F(unary_fixture, Arccos_scl)
{
    check_eq(Arccos::fmap(1.), 0.);
    check_eq(Arccos::bmap(seed, 0., Arccos::fmap(0.)), -seed);
}

TEST_F(unary_fixture, Arccos_vec) 
{
    aVectorXd f = v.acos();
    aVectorXd adj = -seed / (1. - v * v).sqrt();
    aVectorXd vadj = -vseed / (1. - v * v).sqrt();
    test_unary_vec<Arccos>(seed, v, f, adj);
    test_unary_vec<Arccos>(vseed, v, f, vadj);
}

TEST_F(unary_fixture, Arctan_scl)
{
    check_eq(Arctan::fmap(1.), M_PI/4);
    check_eq(Arctan::bmap(seed, 1., M_PI/4), seed * 0.5);
}

TEST_F(unary_fixture, Arctan_vec) 
{
    aVectorXd f = v.atan();
    aVectorXd adj = seed / (1. + v * v);
    aVectorXd vadj = vseed / (1. + v * v);
    test_unary_vec<Arctan>(seed, v, f, adj);
    test_unary_vec<Arctan>(vseed, v, f, vadj);
}

TEST_F(unary_fixture, Exp_scl)
{
    check_eq(Exp::fmap(0.), 1.);
    check_eq(Exp::bmap(seed, 1., Exp::fmap(1.)), seed * std::exp(1.));
}

TEST_F(unary_fixture, Exp_vec) 
{
    aVectorXd f = v.exp();
    aVectorXd adj = seed * f;
    aVectorXd vadj = vseed * f;
    test_unary_vec<Exp>(seed, v, f, adj);
    test_unary_vec<Exp>(vseed, v, f, vadj);
}

TEST_F(unary_fixture, Log_scl)
{
    check_eq(Log::fmap(1.), 0.);
    check_eq(Log::bmap(seed, 2., Log::fmap(2.)), seed * 0.5);
}

TEST_F(unary_fixture, Log_vec) 
{
    aVectorXd f = v.log();
    aVectorXd adj = seed / v;
    aVectorXd vadj = vseed / v;
    test_unary_vec<Log>(seed, v, f, adj);
    test_unary_vec<Log>(vseed, v, f, vadj);
}

TEST_F(unary_fixture, Sqrt_scl)
{
    check_eq(Sqrt::fmap(4.), 2.);
    check_eq(Sqrt::bmap(seed, 4., 2.), seed * 0.25);
}

TEST_F(unary_fixture, Sqrt_vec) 
{
    v(1) = 0.1; // make well-defined
    aVectorXd f = v.sqrt();
    aVectorXd adj = seed / (2 * f);
    aVectorXd vadj = vseed / (2 * f);
    test_unary_vec<Sqrt>(seed, v, f, adj);
    test_unary_vec<Sqrt>(vseed, v, f, vadj);
}

TEST_F(unary_fixture, Erf_scl)
{
    check_eq(Erf::fmap(4.), 0.9999999845827420);
    check_eq(Erf::bmap(seed, 4., Erf::fmap(4.)), seed * 1.2698234671866558e-7);
}

TEST_F(unary_fixture, Erf_vec) 
{
    aVectorXd f = v.erf();
    aVectorXd adj = 2. / std::sqrt(M_PI) * seed * (-v*v).exp();
    aVectorXd vadj = 2. / std::sqrt(M_PI) * vseed * (-v*v).exp();
    test_unary_vec<Erf>(seed, v, f, adj);
    test_unary_vec<Erf>(vseed, v, f, vadj);
}

////////////////////////////////////////////////////////////
// Unary Constant Overloads
////////////////////////////////////////////////////////////

TEST_F(unary_fixture, constant_operator_unary_minus)
{
    test_constant_unary([](const auto& x) {return -x;}, 
                        [](const auto& x) {return -x;});
}

TEST_F(unary_fixture, constant_sin)
{
    test_constant_unary([](const auto& x) {return ad::sin(x);}, 
                        [](const auto& x) {return std::sin(x);});
}

TEST_F(unary_fixture, constant_cos)
{
    test_constant_unary([](const auto& x) {return ad::cos(x);}, 
                        [](const auto& x) {return std::cos(x);});
}

TEST_F(unary_fixture, constant_tan)
{
    test_constant_unary([](const auto& x) {return ad::tan(x);}, 
                        [](const auto& x) {return std::tan(x);});
}

TEST_F(unary_fixture, constant_asin)
{
    test_constant_unary([](const auto& x) {return ad::asin(x);}, 
                        [](const auto& x) {return std::asin(x);});
}

TEST_F(unary_fixture, constant_acos)
{
    test_constant_unary([](const auto& x) {return ad::acos(x);}, 
                        [](const auto& x) {return std::acos(x);});
}

TEST_F(unary_fixture, constant_atan)
{
    test_constant_unary([](const auto& x) {return ad::atan(x);}, 
                        [](const auto& x) {return std::atan(x);});
}

TEST_F(unary_fixture, constant_exp)
{
    test_constant_unary([](const auto& x) {return ad::exp(x);}, 
                        [](const auto& x) {return std::exp(x);});
}

TEST_F(unary_fixture, constant_log)
{
    test_constant_unary([](const auto& x) {return ad::log(x);}, 
                        [](const auto& x) {return std::log(x);});
}

} // namespace core
} // namespace ad
