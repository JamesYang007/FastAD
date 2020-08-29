#include <testutil/base_fixture.hpp>
#include <fastad_bits/reverse/stat/bernoulli.hpp>

namespace ad {
namespace stat {

struct bernoulli_fixture : base_fixture
{
protected:
    using disc_t = int;
    using scl_disc_expr_t = Var<disc_t, scl>;
    using vec_disc_expr_t = Var<disc_t, vec>;
    using scl_disc_expr_view_t = VarView<disc_t, scl>;
    using vec_disc_expr_view_t = VarView<disc_t, vec>;

    using ss_bernoulli_t = BernoulliAdjLogPDFNode<
        scl_disc_expr_view_t, 
        scl_expr_view_t>;

    using vs_bernoulli_t = BernoulliAdjLogPDFNode<
        vec_disc_expr_view_t, 
        scl_expr_view_t>;

    using vv_bernoulli_t = BernoulliAdjLogPDFNode<
        vec_disc_expr_view_t, 
        vec_expr_view_t>;

    scl_disc_expr_t scl_x;
    scl_expr_t scl_p;
    vec_disc_expr_t vec_x;
    vec_expr_t vec_p;

    ss_bernoulli_t ss_bernoulli;
    vs_bernoulli_t vs_bernoulli;
    vv_bernoulli_t vv_bernoulli;

    value_t tol = 1e-15;

    bernoulli_fixture()
        : base_fixture()
        , vec_x(3)
        , vec_p(3)
        , ss_bernoulli(scl_x, scl_p)
        , vs_bernoulli(vec_x, scl_p)
        , vv_bernoulli(vec_x, vec_p)
    {
        // initialize some values
        scl_x.get() = 0;
        scl_p.get() = 0.0001;

        vec_x.get(0,0) = 1;
        vec_x.get(1,0) = 0;
        vec_x.get(2,0) = 1;

        vec_p.get(0,0) = 0.3;
        vec_p.get(1,0) = 0.42;
        vec_p.get(2,0) = 0.98;
    }
};

TEST_F(bernoulli_fixture, ss_feval)
{
    bind(ss_bernoulli);
    value_t res = ss_bernoulli.feval();
    EXPECT_NEAR(res, -0.0001000050003334, tol);
}

TEST_F(bernoulli_fixture, ss_x_one_feval)
{
    scl_x.get() = 1;
    bind(ss_bernoulli);
    value_t res = ss_bernoulli.feval();
    EXPECT_NEAR(res, std::log(scl_p.get()), tol);
}

TEST_F(bernoulli_fixture, ss_feval_p_below_range)
{
    scl_p.get() = 0;
    bind(ss_bernoulli);
    value_t res = ss_bernoulli.feval();
    EXPECT_DOUBLE_EQ(res, 0);
}

TEST_F(bernoulli_fixture, ss_feval_p_above_range)
{
    scl_p.get() = 1.01;
    bind(ss_bernoulli);
    value_t res = ss_bernoulli.feval();
    EXPECT_DOUBLE_EQ(res, util::neg_inf<value_t>);
}

TEST_F(bernoulli_fixture, ss_feval_x_out_of_range)
{
    scl_x.get() = 2;
    bind(ss_bernoulli);
    value_t res = ss_bernoulli.feval();
    EXPECT_DOUBLE_EQ(res, util::neg_inf<value_t>);
}

TEST_F(bernoulli_fixture, ss_beval)
{
    bind(ss_bernoulli);
    ss_bernoulli.feval();
    ss_bernoulli.beval(1);
    EXPECT_NEAR(scl_p.get_adj(), 
                -1.00010001000100001711, 
                tol);
}

TEST_F(bernoulli_fixture, ss_x_one_beval)
{
    scl_x.get() = 1;
    bind(ss_bernoulli);
    ss_bernoulli.feval();
    ss_bernoulli.beval(1);
    EXPECT_NEAR(scl_p.get_adj(), 
                10000.00000000000000000000, 
                tol);
}

TEST_F(bernoulli_fixture, ss_beval_p_below_range)
{
    scl_p.get() = -0.00001;
    bind(ss_bernoulli);
    ss_bernoulli.feval();
    ss_bernoulli.beval(1);
    EXPECT_DOUBLE_EQ(scl_p.get_adj(), 0.);
}

TEST_F(bernoulli_fixture, ss_beval_p_above_range)
{
    scl_p.get() = 1.00001;
    bind(ss_bernoulli);
    ss_bernoulli.feval();
    ss_bernoulli.beval(1);
    EXPECT_DOUBLE_EQ(scl_p.get_adj(), 0.);
}

TEST_F(bernoulli_fixture, ss_beval_x_out_of_range)
{
    scl_x.get() = -1;
    bind(ss_bernoulli);
    ss_bernoulli.feval();
    ss_bernoulli.beval(1);
    EXPECT_DOUBLE_EQ(scl_p.get_adj(), 0.);
}

TEST_F(bernoulli_fixture, vs_feval)
{
    bind(vs_bernoulli);
    value_t res = vs_bernoulli.feval();
    EXPECT_DOUBLE_EQ(res, -18.42078074895269779176);
}

TEST_F(bernoulli_fixture, vs_feval_p_below_range)
{
    scl_p.get() = -0.341;
    bind(vs_bernoulli);
    value_t res = vs_bernoulli.feval();
    EXPECT_DOUBLE_EQ(res, util::neg_inf<value_t>);
}

TEST_F(bernoulli_fixture, vs_feval_p_above_range)
{
    scl_p.get() = 1.341;
    bind(vs_bernoulli);
    value_t res = vs_bernoulli.feval();
    EXPECT_DOUBLE_EQ(res, util::neg_inf<value_t>);
}

TEST_F(bernoulli_fixture, vs_feval_one_x_out_of_range)
{
    vec_x.get(0,0) = 2;
    bind(vs_bernoulli);
    value_t res = vs_bernoulli.feval();
    EXPECT_DOUBLE_EQ(res, util::neg_inf<value_t>);
}

TEST_F(bernoulli_fixture, vs_beval)
{
    bind(vs_bernoulli);
    vs_bernoulli.feval();
    vs_bernoulli.beval(1);
    EXPECT_DOUBLE_EQ(scl_p.get_adj(), 
                     19998.99989998999808449298);
}

TEST_F(bernoulli_fixture, vs_beval_p_below_range)
{
    scl_p.get() = -0.321;
    bind(vs_bernoulli);
    vs_bernoulli.feval();
    vs_bernoulli.beval(1);
    EXPECT_DOUBLE_EQ(scl_p.get_adj(), 0);
}

TEST_F(bernoulli_fixture, vs_beval_p_above_range)
{
    scl_p.get() = 1.321;
    bind(vs_bernoulli);
    vs_bernoulli.feval();
    vs_bernoulli.beval(1);
    EXPECT_DOUBLE_EQ(scl_p.get_adj(), 0);
}

TEST_F(bernoulli_fixture, vs_beval_one_x_out_of_range)
{
    vec_x.get(0,0) = 2;
    bind(vs_bernoulli);
    vs_bernoulli.feval();
    vs_bernoulli.beval(1);
    EXPECT_DOUBLE_EQ(scl_p.get_adj(), 0);
}

TEST_F(bernoulli_fixture, vv_feval)
{
    bind(vv_bernoulli);
    value_t res = vv_bernoulli.feval();
    EXPECT_DOUBLE_EQ(res, -1.76890268708512765627);
}

TEST_F(bernoulli_fixture, vv_feval_p_below_range)
{
    vec_p.get(1,0) = -0.03;
    bind(vv_bernoulli);
    value_t res = vv_bernoulli.feval();
    EXPECT_DOUBLE_EQ(res, -1.22417551164345561610);
}

TEST_F(bernoulli_fixture, vv_feval_p_above_range)
{
    vec_p.get(1,0) = 3.023;
    bind(vv_bernoulli);
    value_t res = vv_bernoulli.feval();
    EXPECT_DOUBLE_EQ(res, util::neg_inf<value_t>);
}

TEST_F(bernoulli_fixture, vv_feval_x_out_of_range)
{
    vec_x.get(1,0) = 3;
    bind(vv_bernoulli);
    value_t res = vv_bernoulli.feval();
    EXPECT_DOUBLE_EQ(res, util::neg_inf<value_t>);
}

TEST_F(bernoulli_fixture, vv_beval)
{
    bind(vv_bernoulli);
    vv_bernoulli.feval();
    vv_bernoulli.beval(1);
 
    EXPECT_DOUBLE_EQ(vec_p.get_adj(0,0), 
                     3.33333333333333348136);
    EXPECT_DOUBLE_EQ(vec_p.get_adj(1,0), 
                     -1.72413793103448265143);
    EXPECT_DOUBLE_EQ(vec_p.get_adj(2,0), 
                     1.02040816326530614511);
}

TEST_F(bernoulli_fixture, vv_beval_p_below_range)
{
    vec_p.get(0,0) = -0.03;
    bind(vv_bernoulli);
    vv_bernoulli.feval();
    vv_bernoulli.beval(1);
    EXPECT_DOUBLE_EQ(vec_p.get_adj(0,0), 0);
    EXPECT_DOUBLE_EQ(vec_p.get_adj(1,0), 
                     -1.72413793103448265143);
    EXPECT_DOUBLE_EQ(vec_p.get_adj(2,0), 
                     1.02040816326530614511);
}

TEST_F(bernoulli_fixture, vv_beval_p_above_range)
{
    vec_p.get(1,0) = 3.023;
    bind(vv_bernoulli);
    vv_bernoulli.feval();
    vv_bernoulli.beval(1);
    EXPECT_DOUBLE_EQ(vec_p.get_adj(0,0), 
                     3.33333333333333348136);
    EXPECT_DOUBLE_EQ(vec_p.get_adj(1,0), 0);
    EXPECT_DOUBLE_EQ(vec_p.get_adj(2,0), 
                     1.02040816326530614511);
}

TEST_F(bernoulli_fixture, vv_beval_x_out_of_range)
{
    vec_x.get(1,0) = 3;
    bind(vv_bernoulli);
    vv_bernoulli.feval();
    vv_bernoulli.beval(1);
    EXPECT_DOUBLE_EQ(vec_p.get_adj(0,0), 0);
    EXPECT_DOUBLE_EQ(vec_p.get_adj(1,0), 0);
    EXPECT_DOUBLE_EQ(vec_p.get_adj(2,0), 0);
}

} // namespace stat
} // namespace ad
