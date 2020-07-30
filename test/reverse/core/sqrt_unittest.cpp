#include <testutil/base_fixture.hpp>
#include <fastad_bits/reverse/core/sqrt.hpp>

namespace ad {
namespace core {

struct sqrt_fixture : base_fixture
{
protected:
    using scl_sqrt_t = SqrtNode<scl_expr_view_t>;
    using vec_sqrt_t = SqrtNode<vec_expr_view_t>;

    scl_sqrt_t scl_sqrt;
    vec_sqrt_t vec_sqrt;

    value_t seed = 23.142;

    sqrt_fixture()
        : base_fixture()
        , scl_sqrt{scl_expr}
        , vec_sqrt{vec_expr}
    {
        vec_expr.get(0,0) = 1.;
        vec_expr.get(1,0) = 2.2131;
        vec_expr.get(2,0) = 4.231;
        vec_expr.get(3,0) = 0.249;
        vec_expr.get(4,0) = 9.883;
    }
};

TEST_F(sqrt_fixture, scl_feval)
{
    bind(scl_sqrt);
    value_t res = scl_sqrt.feval();
    EXPECT_DOUBLE_EQ(res, std::sqrt(scl_expr.get()));
}

TEST_F(sqrt_fixture, scl_beval)
{
    bind(scl_sqrt);
    scl_sqrt.feval();
    scl_sqrt.beval(seed, 0, 0, util::beval_policy::single);
    EXPECT_DOUBLE_EQ(scl_expr.get_adj(0,0), 
                     seed/(2.*std::sqrt(scl_expr.get())));
}

TEST_F(sqrt_fixture, vec_feval)
{
    bind(vec_sqrt);
    Eigen::VectorXd res = vec_sqrt.feval();
    for (int i = 0; i < res.size(); ++i) {
        EXPECT_DOUBLE_EQ(res(i), std::sqrt(vec_expr.get(i,0)));
    }
}

TEST_F(sqrt_fixture, vec_beval)
{
    bind(vec_sqrt);
    vec_sqrt.feval();
    vec_sqrt.beval(seed, 3, 0, util::beval_policy::single);
    for (size_t i = 0; i < vec_expr.size(); ++i) {
        value_t actual = (i == 3) ? 
            seed/(2. * std::sqrt(vec_expr.get(i,0))) : 0;
        EXPECT_DOUBLE_EQ(vec_expr.get_adj(i,0), 
                         actual);
    }
}

} // namespace core
} // namespace ad
