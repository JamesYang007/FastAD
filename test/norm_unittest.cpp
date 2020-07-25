#include <fastad_bits/unary.hpp>
#include <fastad_bits/constant.hpp>
#include <fastad_bits/norm.hpp>
#include "base_fixture.hpp"

namespace ad {
namespace core {

struct norm_fixture : base_fixture
{
protected:
    using unary_t = MockUnary;
    using vec_unary_t = UnaryNode<unary_t, vec_expr_view_t>;
    using norm_t = VecNormNode<vec_unary_t>;

    value_t seed = 9.2313;
    norm_t vec_norm;

    norm_fixture()
        : base_fixture()
        , vec_norm{vec_expr}
    {
        this->bind(vec_norm);
    }
};

TEST_F(norm_fixture, vec_expr_feval)
{
    value_t res = vec_norm.feval();
    value_t actual = 0;
    for (size_t i = 0; i < vec_expr.size(); ++i) {
        value_t val = unary_t::fmap(vec_expr.get(i,0));
        actual += val * val;
    }
    EXPECT_DOUBLE_EQ(res, actual);
}

TEST_F(norm_fixture, vec_expr_beval)
{
    vec_norm.feval();
    vec_norm.beval(seed, 0,0, util::beval_policy::single);    // last two ignored
    for (size_t i = 0; i < vec_expr.size(); ++i) {
        EXPECT_DOUBLE_EQ(vec_expr.get_adj(i,0), 
                seed * 2. * 2. * unary_t::fmap(vec_expr.get(i,0)));
    }
}

TEST_F(norm_fixture, vec_expr_constant)
{
    auto normnode = ad::norm(ad::constant(Eigen::VectorXd(vec_expr.get())));
    static_assert(std::is_same_v<
            std::decay_t<decltype(normnode)>,
            Constant<double, ad::scl> >);
    auto res = normnode.feval();
    value_t actual = 0;
    for (size_t i = 0; i < vec_expr.size(); ++i) {
        value_t val = vec_expr.get(i,0);
        actual += val * val;
    }
    EXPECT_DOUBLE_EQ(res, actual);
}

} // namespace core
} // namespace ad
