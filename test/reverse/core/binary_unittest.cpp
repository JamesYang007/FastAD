#include "gtest/gtest.h"
#include <fastad_bits/reverse/core/binary.hpp>
#include <testutil/base_fixture.hpp>

namespace ad {
namespace core {

struct binary_fixture : base_fixture
{
protected:
    using comp_t = Equal;
    using binary_t = MockBinary;

    using scl_scl_comp_t = 
        BinaryNode<comp_t,
                   scl_expr_view_t,
                   scl_expr_view_t>;
    using vec_scl_comp_t =
        BinaryNode<comp_t,
                   vec_expr_view_t,
                   scl_expr_view_t>;

    using scl_scl_binary_t = 
        BinaryNode<binary_t, 
                   scl_expr_view_t, 
                   scl_expr_view_t>;
    using scl_scl_scl_binary_t = 
        BinaryNode<binary_t, 
                   scl_expr_view_t, 
                   scl_scl_binary_t>;
    using scl_vec_binary_t = 
        BinaryNode<binary_t, 
                   scl_expr_view_t,
                   vec_expr_view_t>;
    using vec_vec_binary_t = 
        BinaryNode<binary_t,
                   vec_expr_view_t,
                   vec_expr_view_t>;
    using mat_mat_binary_t = 
        BinaryNode<binary_t, 
                   mat_expr_view_t,
                   mat_expr_view_t>;

    scl_scl_comp_t scl_scl_comp;
    vec_scl_comp_t vec_scl_comp;

    scl_scl_binary_t scl_scl_binary;
    scl_vec_binary_t scl_vec_binary;
    vec_vec_binary_t vec_vec_binary;
    mat_mat_binary_t mat_mat_binary;
    scl_scl_scl_binary_t scl_scl_scl_binary;

    aVectorXd v;
    aVectorXd u;
    aVectorXd vseed;
    Eigen::ArrayXXd mseed;
    value_t seed = 3.14;

    Eigen::VectorXd val_buf;
    Eigen::VectorXd adj_buf;

    binary_fixture()
        : base_fixture()
        , scl_scl_comp(scl_expr, scl_expr)
        , vec_scl_comp(vec_expr, scl_expr)
        , scl_scl_binary(scl_expr, scl_expr)
        , scl_vec_binary(scl_expr, vec_expr)
        , vec_vec_binary(vec_expr, vec_expr)
        , mat_mat_binary(mat_expr, mat_expr)
        , scl_scl_scl_binary(scl_expr, scl_scl_binary)
        , v(vec_expr.size())
        , u(vec_expr.size())
        , vseed(vec_expr.size())
        , mseed(mat_expr.rows(), mat_expr.cols())
        , val_buf(std::max(vec_size, mat_size))
        , adj_buf(std::max(vec_size, mat_size))
    {
        vseed << 2.13, 0.3231, 4.231, 0.2, 3.21;
        mseed << 2, 3, 4,
                 -1., 2.3, 0.3;
        v << 1, 0, 0.3, 5., 2.;
        u << 1, 2, 3, -1., 3.;

        val_buf.setZero();
        adj_buf.setZero();

        // IMPORTANT: bind value for unary nodes.
        // No two unary node expressions can be used in a single test.
        ptr_pack_t ptr_pack(val_buf.data(), adj_buf.data());
        scl_scl_comp.bind_cache(ptr_pack);
        vec_scl_comp.bind_cache(ptr_pack);
        scl_scl_binary.bind_cache(ptr_pack);
        scl_vec_binary.bind_cache(ptr_pack);
        vec_vec_binary.bind_cache(ptr_pack);
        mat_mat_binary.bind_cache(ptr_pack);
        scl_scl_scl_binary.bind_cache(ptr_pack);
    }

    template <class Binary, class T>
    void test_binary_vec(const Eigen::ArrayBase<T>& sv_f,
                         const Eigen::ArrayBase<T>& vs_f,
                         const Eigen::ArrayBase<T>& vv_f,
                         value_t sv_ladj,
                         const Eigen::ArrayBase<T>& sv_radj,
                         const Eigen::ArrayBase<T>& vs_ladj,
                         value_t vs_radj,
                         const Eigen::ArrayBase<T>& vv_ladj,
                         const Eigen::ArrayBase<T>& vv_radj)
    {
        aVectorXd f;
        value_t scl_adj;
        aVectorXd vec_adj;

        // test scalar-vector
        f = Binary::fmap(v(0), u);
        check_eq(f, sv_f);
        scl_adj = Binary::blmap(vseed, v(0), u, sv_f);
        vec_adj = Binary::brmap(vseed, v(0), u, sv_f);
        check_eq(scl_adj, sv_ladj);
        check_eq(vec_adj, sv_radj);

        // test vector-scalar
        f = Binary::fmap(v, u(0));
        check_eq(f, vs_f);
        vec_adj = Binary::blmap(vseed, v, u(0), vs_f);
        scl_adj = Binary::brmap(vseed, v, u(0), vs_f);
        check_eq(vec_adj, vs_ladj);
        check_eq(scl_adj, vs_radj);

        // test vector-vector
        f = Binary::fmap(v, u);
        check_eq(f, vv_f);
        vec_adj = Binary::blmap(vseed, v, u, vv_f);
        check_eq(vec_adj, vv_ladj);
        vec_adj = Binary::brmap(vseed, v, u, vv_f);
        check_eq(vec_adj, vv_radj);
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

////////////////////////////////////////////////////////////////////////
// BinaryNode TEST
////////////////////////////////////////////////////////////////////////

TEST_F(binary_fixture, scl_scl_feval)
{
    value_t res = scl_scl_binary.feval();
    EXPECT_DOUBLE_EQ(res, binary_t::fmap(scl_expr.get(), scl_expr.get()));
}

TEST_F(binary_fixture, scl_scl_beval)
{
    scl_scl_binary.beval(seed);
    EXPECT_DOUBLE_EQ(scl_expr.get_adj(), 
                     binary_t::blmap(seed, scl_expr.get(), scl_expr.get(), scl_scl_binary.get()) +
                     binary_t::brmap(seed, scl_expr.get(), scl_expr.get(), scl_scl_binary.get()));
}

TEST_F(binary_fixture, scl_vec_feval)
{
    aVectorXd res = scl_vec_binary.feval();
    aVectorXd actual = binary_t::fmap(scl_expr.get(), vec_expr.get().array()); 
    check_eq(res, actual);
}

TEST_F(binary_fixture, scl_vec_beval)
{
    scl_vec_binary.beval(vseed);
    auto a_v = vec_expr.get().array();
    auto a_sv = scl_vec_binary.get().array();
    aVectorXd actual = binary_t::brmap(vseed, scl_expr.get(), a_v, a_sv);
    EXPECT_DOUBLE_EQ(scl_expr.get_adj(), binary_t::blmap(vseed, scl_expr.get(), a_v, a_sv));
    check_eq(vec_expr.get_adj(), actual);
}

TEST_F(binary_fixture, vec_vec_feval)
{
    aVectorXd res = vec_vec_binary.feval();
    auto a_v = vec_expr.get().array();
    aVectorXd actual = binary_t::fmap(a_v, a_v);
    check_eq(res, actual);
}

TEST_F(binary_fixture, vec_vec_beval)
{
    vec_vec_binary.beval(vseed);
    auto a_v = vec_expr.get().array();
    auto a_vv = vec_vec_binary.get().array();
    aVectorXd actual = 
        binary_t::blmap(vseed, a_v, a_v, a_vv) +
        binary_t::brmap(vseed, a_v, a_v, a_vv);
    check_eq(vec_expr.get_adj(), actual);   
}

TEST_F(binary_fixture, mat_mat_feval)
{
    Eigen::MatrixXd res = mat_mat_binary.feval();
    auto a_m = mat_expr.get().array();
    Eigen::MatrixXd actual = binary_t::fmap(a_m, a_m);
    check_eq(res, actual);
}

TEST_F(binary_fixture, mat_mat_beval)
{
    mat_mat_binary.beval(mseed);
    auto a_m = mat_expr.get().array();
    auto a_mm = mat_mat_binary.get().array();
    Eigen::MatrixXd actual = 
        binary_t::blmap(mseed, a_m, a_m, a_mm) +
        binary_t::brmap(mseed, a_m, a_m, a_mm);
    check_eq(mat_expr.get_adj(), actual);
}

TEST_F(binary_fixture, scl_scl_scl_feval)
{
    value_t res = scl_scl_scl_binary.feval();
    auto& s = scl_expr.get();
    EXPECT_DOUBLE_EQ(res, binary_t::fmap(s, binary_t::fmap(s,s)));
}

TEST_F(binary_fixture, scl_scl_scl_beval)
{
    scl_scl_scl_binary.beval(seed);
    EXPECT_DOUBLE_EQ(scl_expr.get_adj(), 3.*seed);
}

TEST_F(binary_fixture, scl_scl_comp_feval)
{
    EXPECT_TRUE(scl_scl_comp.feval());
}

TEST_F(binary_fixture, scl_scl_comp_beval)
{
    scl_scl_comp.beval(seed);
    EXPECT_DOUBLE_EQ(scl_expr.get_adj(), 0.);
}

TEST_F(binary_fixture, vec_scl_comp_feval)
{
    Eigen::Array<bool, Eigen::Dynamic, 1> actual(vec_expr.size());
    actual = comp_t::fmap(vec_expr.get().array(), scl_expr.get());
    check_eq(vec_scl_comp.feval(), actual);
}

TEST_F(binary_fixture, vec_scl_comp_beval)
{
    vec_scl_comp.beval(seed);
    EXPECT_DOUBLE_EQ(scl_expr.get_adj(), 0.);
    aVectorXd actual(vec_expr.size());
    actual.setZero();
    check_eq(vec_expr.get_adj(), actual);
}

////////////////////////////////////////////////////////////////////////
// Struct TEST
////////////////////////////////////////////////////////////////////////

TEST_F(binary_fixture, Add_scl) 
{
    EXPECT_DOUBLE_EQ(Add::fmap(-1.0, 2.1), 1.1);
    EXPECT_DOUBLE_EQ(Add::blmap(seed, -2.01, 2341.2131, 0), seed);
    EXPECT_DOUBLE_EQ(Add::brmap(seed, -2.01, 2341.2131, 0), seed);
}

TEST_F(binary_fixture, Add_vec) 
{
    aVectorXd sv_f = (v(0) + u);
    aVectorXd vs_f = (v + u(0));
    aVectorXd vv_f = (v + u);
    value_t sv_ladj = vseed.sum();
    aVectorXd sv_radj = vseed;
    test_binary_vec<Add>(sv_f, vs_f, vv_f,
                         sv_ladj, sv_radj,
                         sv_radj, sv_ladj,
                         sv_radj, sv_radj);
}

TEST_F(binary_fixture, Sub_scl) 
{
    EXPECT_DOUBLE_EQ(Sub::fmap(-1.0, 2.1), -3.1);
    EXPECT_DOUBLE_EQ(Sub::blmap(seed, -2.01, 2., 0), seed);
    EXPECT_DOUBLE_EQ(Sub::brmap(seed, -2.01, 3., 0), -seed);
}

TEST_F(binary_fixture, Sub_vec) 
{
    aVectorXd sv_f = (v(0) - u);
    aVectorXd vs_f = (v - u(0));
    aVectorXd vv_f = (v - u);
    value_t sv_ladj = vseed.sum();
    aVectorXd sv_radj = -vseed;
    aVectorXd vs_ladj = -sv_radj;
    value_t vs_radj = -sv_ladj;
    aVectorXd vv_ladj = -sv_radj;
    aVectorXd vv_radj = sv_radj;
    test_binary_vec<Sub>(sv_f, vs_f, vv_f,
                         sv_ladj, sv_radj,
                         vs_ladj, vs_radj,
                         vv_ladj, vv_radj);
}

TEST_F(binary_fixture, Mul_scl) 
{
    EXPECT_DOUBLE_EQ(Mul::fmap(-1.0, 2.1), -2.1);
    EXPECT_DOUBLE_EQ(Mul::blmap(seed, -2.01, 2., 0), 2. * seed);
    EXPECT_DOUBLE_EQ(Mul::brmap(seed, -2.01, 3., 0), -2.01 * seed);
}

TEST_F(binary_fixture, Mul_vec) 
{
    aVectorXd sv_f = (v(0) * u);
    aVectorXd vs_f = (v * u(0));
    aVectorXd vv_f = (v * u);
    value_t sv_ladj = (vseed * u).sum();
    aVectorXd sv_radj = vseed * v(0);
    aVectorXd vs_ladj = vseed * u(0);
    value_t vs_radj = (vseed * v).sum();
    aVectorXd vv_ladj = vseed * u;
    aVectorXd vv_radj = vseed * v;
    test_binary_vec<Mul>(sv_f, vs_f, vv_f,
                         sv_ladj, sv_radj,
                         vs_ladj, vs_radj,
                         vv_ladj, vv_radj);
}

TEST_F(binary_fixture, Div_scl)
{
    EXPECT_DOUBLE_EQ(Div::fmap(-1.0, 2.1), -1./2.1);
    EXPECT_DOUBLE_EQ(Div::blmap(seed, -2.01, 2., -2.01/2.), seed * 0.5);
    EXPECT_DOUBLE_EQ(Div::brmap(seed, -2.01, 3., -2.01/3.), seed * 2.01 / 9.);
}

TEST_F(binary_fixture, Div_vec)
{
    aVectorXd sv_f = (v(0) / u);
    aVectorXd vs_f = (v / u(0));
    aVectorXd vv_f = (v / u);
    value_t sv_ladj = (vseed / u).sum();
    aVectorXd sv_radj = -vseed * v(0) / (u * u);
    aVectorXd vs_ladj = vseed / u(0);
    value_t vs_radj = (-vseed * v / (u(0) * u(0))).sum();
    aVectorXd vv_ladj = vseed / u;
    aVectorXd vv_radj = -vseed * v / (u * u);
    test_binary_vec<Div>(sv_f, vs_f, vv_f,
                         sv_ladj, sv_radj,
                         vs_ladj, vs_radj,
                         vv_ladj, vv_radj);
}

//////////////////////////////////////////////////////////////////
// Binary Constant TEST
//////////////////////////////////////////////////////////////////

TEST_F(binary_fixture, constant_operator_plus)
{
    test_constant_binary<double>([](const auto& x, const auto& y) { return x + y; },
                         [](const auto& x, const auto& y) { return x + y; });
}

TEST_F(binary_fixture, constant_operator_minus)
{
    test_constant_binary<double>([](const auto& x, const auto& y) { return x - y; },
                         [](const auto& x, const auto& y) { return x - y; });
}

TEST_F(binary_fixture, constant_operator_mult)
{
    test_constant_binary<double>([](const auto& x, const auto& y) { return x * y; },
                         [](const auto& x, const auto& y) { return x * y; });
}

TEST_F(binary_fixture, constant_operator_div)
{
    test_constant_binary<double>([](const auto& x, const auto& y) { return x / y; },
                         [](const auto& x, const auto& y) { return x / y; });
}

TEST_F(binary_fixture, constant_operator_less)
{
    test_constant_binary<bool>([](const auto& x, const auto& y) { return x < y; },
                         [](const auto& x, const auto& y) { return x < y; });
}

TEST_F(binary_fixture, constant_operator_less_eq)
{
    test_constant_binary<bool>([](const auto& x, const auto& y) { return x <= y; },
                         [](const auto& x, const auto& y) { return x <= y; });
}

TEST_F(binary_fixture, constant_operator_greater)
{
    test_constant_binary<bool>([](const auto& x, const auto& y) { return x > y; },
                         [](const auto& x, const auto& y) { return x > y; });
}

TEST_F(binary_fixture, constant_operator_greater_than)
{
    test_constant_binary<bool>([](const auto& x, const auto& y) { return x >= y; },
                         [](const auto& x, const auto& y) { return x >= y; });
}

TEST_F(binary_fixture, constant_operator_eq)
{
    test_constant_binary<bool>([](const auto& x, const auto& y) { return x == y; },
                         [](const auto& x, const auto& y) { return x == y; });
}

TEST_F(binary_fixture, constant_operator_neq)
{
    test_constant_binary<bool>([](const auto& x, const auto& y) { return x != y; },
                         [](const auto& x, const auto& y) { return x != y; });
}

TEST_F(binary_fixture, constant_operator_and)
{
    test_constant_binary<bool>([](const auto& x, const auto& y) { return LogicalAnd::fmap(x,y); },
                         [](const auto& x, const auto& y) { return LogicalAnd::fmap(x,y); });
}

TEST_F(binary_fixture, constant_operator_or)
{
    test_constant_binary<bool>([](const auto& x, const auto& y) { return LogicalOr::fmap(x,y); },
                         [](const auto& x, const auto& y) { return LogicalOr::fmap(x,y); });
}

} // namespace core
} // namespace ad
