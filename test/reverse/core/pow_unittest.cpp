#include "gtest/gtest.h"
#include <fastad_bits/reverse/core/pow.hpp>
#include <fastad_bits/reverse/core/unary.hpp>
#include <testutil/base_fixture.hpp>

namespace ad {
namespace core {

struct pow_fixture : base_fixture
{
protected:
    using unary_t = MockUnary;
    using scl_unary_t = UnaryNode<unary_t, scl_expr_view_t>;
    using vec_unary_t = UnaryNode<unary_t, vec_expr_view_t>;
    using mat_unary_t = UnaryNode<unary_t, mat_expr_view_t>;
    using scl_sq_t = PowNode<2, scl_unary_t>;
    using scl_inv_t = PowNode<-1, scl_unary_t>;
    using scl_const_t = PowNode<0, scl_unary_t>;
    using vec_sq_t = PowNode<2, vec_unary_t>;
    using vec_inv_t = PowNode<-1, vec_unary_t>;
    using vec_const_t = PowNode<0, vec_unary_t>;

    scl_sq_t scl_sq;
    scl_inv_t scl_inv;
    scl_const_t scl_const;
    vec_sq_t vec_sq;
    vec_inv_t vec_inv;
    vec_const_t vec_const;

    value_t seed = 0.32188;
    aVectorXd vseed;

    Eigen::VectorXd val_buf;
    Eigen::VectorXd adj_buf;

    pow_fixture()
        : base_fixture()
        , scl_sq{scl_expr}
        , scl_inv{scl_expr}
        , scl_const{scl_expr}
        , vec_sq{vec_expr}
        , vec_inv{vec_expr}
        , vec_const{vec_expr}
        , vseed(vec_expr.size())
        , val_buf(vec_sq.bind_cache_size()(0))
        , adj_buf(vec_sq.bind_cache_size()(1))
    {
        vseed << 2.13, 0.3231, 4.231, 2.41, 3.13;

        ptr_pack_t ptr_pack(val_buf.data(), adj_buf.data());
        scl_sq.bind_cache(ptr_pack);
        scl_inv.bind_cache(ptr_pack);
        scl_const.bind_cache(ptr_pack);
        vec_sq.bind_cache(ptr_pack);
        vec_inv.bind_cache(ptr_pack);
        vec_const.bind_cache(ptr_pack);
    }

    template <class T>
    T sq(T x) { return x * x; }

    template <class T>
    T inv(T x) { 
        return (x == 0) ? 
            std::numeric_limits<T>::infinity() : 
            1./x; 
    }
};

// scl TEST

TEST_F(pow_fixture, scl_sq_feval)
{
    value_t res = scl_sq.feval();
    EXPECT_DOUBLE_EQ(res, sq(2*scl_expr.get()));
}

TEST_F(pow_fixture, scl_sq_beval)
{
    scl_sq.feval();
    scl_sq.beval(seed);
    EXPECT_DOUBLE_EQ(scl_expr.get_adj(), 
                     seed * 8.*scl_expr.get());
}

TEST_F(pow_fixture, scl_sq_x_zero_beval)
{
    scl_expr.get() = 0;
    scl_sq.feval();
    scl_sq.beval(seed);
    EXPECT_DOUBLE_EQ(scl_expr.get_adj(), 0);
}

TEST_F(pow_fixture, scl_inv_feval)
{
    value_t res = scl_inv.feval();
    EXPECT_DOUBLE_EQ(res, inv(2*scl_expr.get()));
}

TEST_F(pow_fixture, scl_inv_beval)
{
    scl_inv.feval();
    scl_inv.beval(seed);
    EXPECT_DOUBLE_EQ(scl_expr.get_adj(), 
                     seed * 2 * -inv(sq(2.*scl_expr.get())));
}

TEST_F(pow_fixture, scl_const_feval)
{
    value_t res = scl_const.feval();
    EXPECT_DOUBLE_EQ(res, 1);
}

TEST_F(pow_fixture, scl_const_beval)
{
    scl_const.feval();
    scl_const.beval(seed);
    EXPECT_DOUBLE_EQ(scl_expr.get_adj(), 0.);
}

// vec TEST

TEST_F(pow_fixture, vec_sq_feval)
{
    Eigen::VectorXd res = vec_sq.feval();
    for (int i = 0; i < res.size(); ++i) {
        EXPECT_DOUBLE_EQ(res(i), sq(2*vec_expr.get(i,0)));
    }
}

TEST_F(pow_fixture, vec_sq_beval)
{
    vec_sq.feval();
    vec_sq.beval(vseed);
    for (size_t i = 0; i < vec_expr.size(); ++i) {
        value_t actual = vseed(i) * 8.* vec_expr.get(i,0);
        EXPECT_DOUBLE_EQ(vec_expr.get_adj(i,0), actual);
    }
}

TEST_F(pow_fixture, vec_inv_feval)
{
    Eigen::VectorXd res = vec_inv.feval();
    for (int i = 0; i < res.size(); ++i) {
        EXPECT_DOUBLE_EQ(res(i), inv(2*vec_expr.get(i,0)));
    }
}

TEST_F(pow_fixture, vec_inv_beval)
{
    vec_inv.feval();
    vec_inv.beval(vseed);
    for (size_t i = 0; i < vec_expr.size(); ++i) {
        value_t actual = vseed(i) * 2 * -inv(sq(2.*vec_expr.get(i,0)));
        EXPECT_DOUBLE_EQ(vec_expr.get_adj(i,0), actual);
    }
}

TEST_F(pow_fixture, vec_const_feval)
{
    Eigen::VectorXd res = vec_const.feval();
    for (int i = 0; i < res.size(); ++i) {
        EXPECT_DOUBLE_EQ(res(i), 1);
    }
}

TEST_F(pow_fixture, vec_const_beval)
{
    vec_const.feval();
    vec_const.beval(vseed);
    for (size_t i = 0; i < vec_expr.size(); ++i) {
        EXPECT_DOUBLE_EQ(vec_expr.get_adj(i,0), 0.);
    }
}

} // namespace core
} // namespace ad
