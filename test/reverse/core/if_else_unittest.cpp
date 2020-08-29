#include <array>
#include <testutil/base_fixture.hpp>
#include <fastad_bits/reverse/core/if_else.hpp>
#include <fastad_bits/reverse/core/binary.hpp>

namespace ad {
namespace core {

struct if_else_fixture : base_fixture
{
protected:
    using scl_constant_t = Constant<value_t, scl>;

    scl_expr_t x{1.}, y{2.}, z{3.}, w{4.};
    vec_expr_t vx, vy, vz, vw;
    std::array<scl_constant_t, 3> const_exprs = {0., 3., 4.};
    std::vector<value_t> val_buf;

    if_else_fixture()
        : base_fixture()
        , vx(vec_size)
        , vy(vec_size)
        , vz(vec_size)
        , vw(vec_size)
    {
        this->vec_initialize(vx);
        this->vec_initialize(vy);
        this->vec_initialize(vz);
        this->vec_initialize(vw);
    }

    template <class ExprType>
    void bind(ExprType& expr) 
    {
        val_buf.resize(expr.bind_cache_size()(0));
        adj_buf.resize(expr.bind_cache_size()(1));
        expr.bind_cache({val_buf.data(), adj_buf.data()});
    }
};

TEST_F(if_else_fixture, cond_expr_simple)
{
    auto expr = (x < y); 
    bind(expr);
    bool cond = expr.feval();
    EXPECT_TRUE(cond);
}

TEST_F(if_else_fixture, cond_expr_two_cond)
{
    auto expr = (x < y) || (z >= w); 
    bind(expr);
    bool cond = expr.feval();
    EXPECT_TRUE(cond);
}

TEST_F(if_else_fixture, cond_expr_three_cond)
{
    auto expr = (x > y) || ((z >= w) && (x == z)); 
    bind(expr);
    bool cond = expr.feval();
    EXPECT_FALSE(cond);
}

TEST_F(if_else_fixture, if_else_simple)
{
    auto expr = if_else(x < y, x, y);
    bind(expr);
    double value = expr.feval();
    EXPECT_DOUBLE_EQ(value, 1.);
    expr.beval(1.);
    EXPECT_DOUBLE_EQ(x.get_adj(0,0), 1.); // updated
    EXPECT_DOUBLE_EQ(y.get_adj(0,0), 0.); // unchanged
}

TEST_F(if_else_fixture, if_else_simple_negated)
{
    auto expr = if_else(x >= y, x, y);
    bind(expr);
    double value = expr.feval();
    EXPECT_DOUBLE_EQ(value, 2.);
    expr.beval(1.);
    EXPECT_DOUBLE_EQ(y.get_adj(0,0), 1.); // updated
    EXPECT_DOUBLE_EQ(x.get_adj(0,0), 0.); // unchanged
}

TEST_F(if_else_fixture, if_else_complicated)
{
    auto expr = if_else(
        (x < y) && (z < w),
        x * y + z,
        x);
    bind(expr);
    double value = expr.feval();
    EXPECT_DOUBLE_EQ(value, 5.);
    expr.beval(1.);
    EXPECT_DOUBLE_EQ(x.get_adj(0,0), 2.);
    EXPECT_DOUBLE_EQ(y.get_adj(0,0), 1.);
    EXPECT_DOUBLE_EQ(z.get_adj(0,0), 1.);
}

TEST_F(if_else_fixture, if_else_complicated_vec)
{
    auto expr = if_else(
        (x < y) && (z < w),
        vx * vy + vz,
        vx);
    bind(expr);
    Eigen::VectorXd res = expr.feval();

    for (int i = 0; i < res.size(); ++i) {
        EXPECT_DOUBLE_EQ(res(i), 
                vx.get(i,0) * vy.get(i,0) + vz.get(i,0));
    }

    expr.beval(1.);

    for (size_t i = 0; i < vec_size; ++i) {
        EXPECT_DOUBLE_EQ(vx.get_adj(i,0), vy.get(i,0));
        EXPECT_DOUBLE_EQ(vy.get_adj(i,0), vx.get(i,0));
        EXPECT_DOUBLE_EQ(vz.get_adj(i,0), 1.);
    }
}

TEST_F(if_else_fixture, if_on_if)
{
    auto expr = 
        if_else(
            (x < y),
            if_else(
                z < w,
                x * y + z,
                x
            ),
            ad::constant(0.)
        );
    bind(expr);
    double value = expr.feval();
    EXPECT_DOUBLE_EQ(value, 5.);

    expr.beval(1.);
    EXPECT_DOUBLE_EQ(x.get_adj(0,0), 2.);
    EXPECT_DOUBLE_EQ(y.get_adj(0,0), 1.);
    EXPECT_DOUBLE_EQ(z.get_adj(0,0), 1.);

    x.reset_adj();
    y.reset_adj();
    z.reset_adj();

    expr.beval(1.);
    EXPECT_DOUBLE_EQ(x.get_adj(0,0), 2.);
    EXPECT_DOUBLE_EQ(y.get_adj(0,0), 1.);
    EXPECT_DOUBLE_EQ(z.get_adj(0,0), 1.);
}

TEST_F(if_else_fixture, if_constants)
{
    auto expr = if_else(const_exprs[0],
                        const_exprs[1],
                        const_exprs[2].get());
    static_assert(std::is_same_v<
            std::decay_t<decltype(expr)>,
            Constant<double, scl> >);
    EXPECT_DOUBLE_EQ(expr.feval(), 4.);
}

} // namespace core
} // namespace ad
