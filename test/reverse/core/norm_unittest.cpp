#include <testutil/base_fixture.hpp>
#include <fastad_bits/reverse/core/unary.hpp>
#include <fastad_bits/reverse/core/constant.hpp>
#include <fastad_bits/reverse/core/norm.hpp>

namespace ad {
namespace core {

struct norm_fixture : base_fixture
{
protected:
    using unary_t = MockUnary;
    using vec_unary_t = UnaryNode<unary_t, vec_expr_view_t>;
    using mat_unary_t = UnaryNode<unary_t, mat_expr_view_t>;
    using norm_t = NormNode<vec_unary_t>;
    using mat_norm_t = NormNode<mat_unary_t>;

    value_t seed = 9.2313;
    norm_t vec_norm;
    mat_norm_t mat_norm;

    norm_fixture()
        : base_fixture()
        , vec_norm{vec_expr}
        , mat_norm{mat_expr}
    {
        this->bind(vec_norm);
    }
};

TEST_F(norm_fixture, vec_expr_feval)
{
    value_t res = vec_norm.feval();
    value_t actual = 
        unary_t::fmap(vec_expr.get().array())
            .matrix().squaredNorm();
    EXPECT_DOUBLE_EQ(res, actual);
}

TEST_F(norm_fixture, vec_expr_beval)
{
    Eigen::VectorXd uvec = unary_t::fmap(vec_expr.get().array());
    Eigen::VectorXd adj = unary_t::bmap(
        seed * 2 * uvec.array(),
        vec_expr.get().array(),
        uvec.array());
    vec_norm.feval();
    vec_norm.beval(seed);
    check_eq(adj, vec_expr.get_adj());
}

TEST_F(norm_fixture, mat_expr_feval)
{
    this->bind(mat_norm);
    value_t res = mat_norm.feval();
    value_t actual = 
        unary_t::fmap(mat_expr.get().array())
            .matrix().squaredNorm();
    EXPECT_DOUBLE_EQ(res, actual);
}

TEST_F(norm_fixture, mat_expr_beval)
{
    Eigen::MatrixXd umat = unary_t::fmap(mat_expr.get().array());
    Eigen::MatrixXd adj = unary_t::bmap(
        seed * 2 * umat.array(),
        mat_expr.get().array(),
        umat.array());

    this->bind(mat_norm);
    mat_norm.feval();
    mat_norm.beval(seed);
    check_eq(mat_expr.get_adj(), adj);
}

TEST_F(norm_fixture, vec_expr_constant)
{
    auto normnode = ad::norm(ad::constant(vec_expr.get()));
    static_assert(std::is_same_v<
            std::decay_t<decltype(normnode)>,
            Constant<double, ad::scl> >);
    auto res = normnode.feval();
    value_t actual = vec_expr.get().squaredNorm();
    EXPECT_DOUBLE_EQ(res, actual);
}

} // namespace core
} // namespace ad
