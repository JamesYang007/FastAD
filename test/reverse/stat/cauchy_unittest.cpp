#include <testutil/base_fixture.hpp>
#include <fastad_bits/reverse/stat/cauchy.hpp>

namespace ad {
namespace stat {

struct cauchy_fixture : base_fixture
{
protected:
    using sss_cauchy_t = CauchyAdjLogPDFNode<
        scl_expr_view_t, 
        scl_expr_view_t, 
        scl_expr_view_t>;

    using vss_cauchy_t = CauchyAdjLogPDFNode<
        vec_expr_view_t, 
        scl_expr_view_t, 
        scl_expr_view_t>;

    using vsv_cauchy_t = CauchyAdjLogPDFNode<
        vec_expr_view_t, 
        scl_expr_view_t, 
        vec_expr_view_t>;

    using vvs_cauchy_t = CauchyAdjLogPDFNode<
        vec_expr_view_t, 
        vec_expr_view_t, 
        scl_expr_view_t>;

    using vvv_cauchy_t = CauchyAdjLogPDFNode<
        vec_expr_view_t, 
        vec_expr_view_t, 
        vec_expr_view_t>;

    scl_expr_t scl_x;
    scl_expr_t scl_loc;
    scl_expr_t scl_scale;
    vec_expr_t vec_x;
    vec_expr_t vec_loc;
    vec_expr_t vec_scale;

    sss_cauchy_t sss_cauchy;
    vss_cauchy_t vss_cauchy;
    vsv_cauchy_t vsv_cauchy;
    vvs_cauchy_t vvs_cauchy;
    vvv_cauchy_t vvv_cauchy;

    value_t tol = 1e-15;

    cauchy_fixture()
        : base_fixture()
        , vec_x(3)
        , vec_loc(3)
        , vec_scale(3)
        , sss_cauchy(scl_x, scl_loc, scl_scale)
        , vss_cauchy(vec_x, scl_loc, scl_scale)
        , vsv_cauchy(vec_x, scl_loc, vec_scale)
        , vvs_cauchy(vec_x, vec_loc, scl_scale)
        , vvv_cauchy(vec_x, vec_loc, vec_scale)
    {
        // initialize some values
        scl_x.get() = 0.421;
        scl_loc.get() = 0.341;
        scl_scale.get() = 2.132;

        vec_x.get(0,0) = 0.5;
        vec_x.get(1,0) = -1.3;
        vec_x.get(2,0) = -3.2414999;

        vec_loc.get(0,0) = 0.4;
        vec_loc.get(1,0) = -2.30000001;
        vec_loc.get(2,0) = -10.32;

        vec_scale.get(0,0) = 0.51;
        vec_scale.get(1,0) = 0.01;
        vec_scale.get(2,0) = 3.4;
    }
};

TEST_F(cauchy_fixture, sss_feval)
{
    bind(sss_cauchy);
    value_t res = sss_cauchy.feval();
    EXPECT_DOUBLE_EQ(res, -0.7584675254495730);
}

TEST_F(cauchy_fixture, sss_feval_out_of_range)
{
    scl_scale.get() = 0.0;
    bind(sss_cauchy);
    value_t res = sss_cauchy.feval();
    EXPECT_DOUBLE_EQ(res, util::neg_inf<value_t>);
}

TEST_F(cauchy_fixture, sss_beval)
{
    bind(sss_cauchy);
    sss_cauchy.feval();
    sss_cauchy.beval(1);
    EXPECT_DOUBLE_EQ(scl_x.get_adj(0,0),
                     -0.0351507439654960);
    EXPECT_DOUBLE_EQ(scl_loc.get_adj(0,0), 
                     -scl_x.get_adj(0,0));
    EXPECT_DOUBLE_EQ(scl_scale.get_adj(0,0), 
                     -0.4677241747104879);
}

TEST_F(cauchy_fixture, vss_feval)
{
    bind(vss_cauchy);
    value_t res = vss_cauchy.feval();
    EXPECT_DOUBLE_EQ(res, -4.0831775617990589);
}

TEST_F(cauchy_fixture, vss_feval_out_of_range)
{
    scl_scale.get(0,0) = -0.0000001;
    bind(vss_cauchy);
    value_t res = vss_cauchy.feval();
    EXPECT_DOUBLE_EQ(res, util::neg_inf<value_t>);
}

TEST_F(cauchy_fixture, vss_beval)
{
    bind(vss_cauchy);
    vss_cauchy.feval();
    vss_cauchy.beval(1);
    EXPECT_DOUBLE_EQ(vec_x.get_adj(0,0),
                     -0.0695735121824751);
    EXPECT_DOUBLE_EQ(vec_x.get_adj(1,0),
                     0.4534210702643782);
    EXPECT_DOUBLE_EQ(vec_x.get_adj(2,0),
                     0.4122618701395337);
    EXPECT_DOUBLE_EQ(scl_loc.get_adj(0,0), 
                     -0.7961094282214367);
    EXPECT_DOUBLE_EQ(scl_scale.get_adj(0,0), 
                     -0.3601996841981470);
}

TEST_F(cauchy_fixture, vsv_feval)
{
    bind(vsv_cauchy);
    vsv_cauchy.feval();
    value_t res = vsv_cauchy.feval();
    EXPECT_DOUBLE_EQ(res, -6.9858076420069022);
}

TEST_F(cauchy_fixture, vsv_feval_out_of_range)
{
    vec_scale.get(0,0) = -0.0001;
    bind(vsv_cauchy);
    vsv_cauchy.feval();
    value_t res = vsv_cauchy.feval();
    EXPECT_DOUBLE_EQ(res, util::neg_inf<value_t>);
}

TEST_F(cauchy_fixture, vsv_beval)
{
    bind(vsv_cauchy);
    vsv_cauchy.feval();
    vsv_cauchy.beval(1);
    
    EXPECT_DOUBLE_EQ(vec_x.get_adj(0,0),
                     -1.1142998307525727);
    EXPECT_DOUBLE_EQ(vec_x.get_adj(1,0),
                     1.2187237860200275);
    EXPECT_DOUBLE_EQ(vec_x.get_adj(2,0),
                     0.2937160801794709);

    EXPECT_DOUBLE_EQ(scl_loc.get_adj(0,0), 
                     -0.3981400354469258);

    EXPECT_DOUBLE_EQ(vec_scale.get_adj(0,0), 
                     -1.6133849547261585);
    EXPECT_DOUBLE_EQ(vec_scale.get_adj(1,0), 
                     99.9925732858865359);
    EXPECT_NEAR(vec_scale.get_adj(2,0), 
                0.0153640670209843,
                tol);
}

TEST_F(cauchy_fixture, vvs_feval)
{
    bind(vvs_cauchy);
    vvs_cauchy.feval();
    value_t res = vvs_cauchy.feval();
    EXPECT_DOUBLE_EQ(res, -4.959070146586642025);
}

TEST_F(cauchy_fixture, vvs_feval_out_of_range)
{
    scl_scale.get(0,0) = -0.03234;
    bind(vvs_cauchy);
    vvs_cauchy.feval();
    value_t res = vvs_cauchy.feval();
    EXPECT_DOUBLE_EQ(res, util::neg_inf<value_t>);
}

TEST_F(cauchy_fixture, vvs_beval)
{
    bind(vvs_cauchy);
    vvs_cauchy.feval();
    vvs_cauchy.beval(1);

    EXPECT_DOUBLE_EQ(vec_x.get_adj(0,0),
                     -0.043903706877779093);
    EXPECT_DOUBLE_EQ(vec_x.get_adj(1,0),
                     -0.360657726584449723);
    EXPECT_DOUBLE_EQ(vec_x.get_adj(2,0),
                     -0.259045708467015523);

    for (size_t i = 0; i < vec_x.size(); ++i) {
        EXPECT_DOUBLE_EQ(vec_loc.get_adj(i,0), 
                         -vec_x.get_adj(i,0));
    }

    EXPECT_DOUBLE_EQ(scl_scale.get_adj(0,0), 
                     -0.375842788852183252);

}

TEST_F(cauchy_fixture, vvv_feval)
{
    bind(vvv_cauchy);
    vvv_cauchy.feval();
    value_t res = vvv_cauchy.feval();
    EXPECT_DOUBLE_EQ(res, -6.867595467569049816);
}

TEST_F(cauchy_fixture, vvv_feval_out_of_range)
{
    vec_scale.get(1,0) = 0.0;
    bind(vvv_cauchy);
    vvv_cauchy.feval();
    value_t res = vvv_cauchy.feval();
    EXPECT_DOUBLE_EQ(res, util::neg_inf<value_t>);
}

TEST_F(cauchy_fixture, vvv_beval)
{
    bind(vvv_cauchy);
    vvv_cauchy.feval();
    vvv_cauchy.beval(1);
    
    EXPECT_DOUBLE_EQ(vec_x.get_adj(0,0),
                     -0.740466493891151267);
    EXPECT_DOUBLE_EQ(vec_x.get_adj(1,0),
                     -1.999800000003999267);
    EXPECT_DOUBLE_EQ(vec_x.get_adj(2,0),
                     -0.229578571732138997);

    for (size_t i = 0; i < vec_x.size(); ++i) {
        EXPECT_DOUBLE_EQ(vec_loc.get_adj(i,0), 
                         -vec_x.get_adj(i,0));
    }

    EXPECT_DOUBLE_EQ(vec_scale.get_adj(0,0), 
                     -1.815594805119381983);
    EXPECT_DOUBLE_EQ(vec_scale.get_adj(1,0), 
                     99.980002000199945655);
    EXPECT_DOUBLE_EQ(vec_scale.get_adj(2,0), 
                     0.183844689107000858);
}

} // namespace stat
} // namespace ad
