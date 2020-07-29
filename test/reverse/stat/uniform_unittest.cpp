#include <testutil/base_fixture.hpp>
#include <fastad_bits/reverse/stat/uniform.hpp>

namespace ad {
namespace core {

struct uniform_fixture : base_fixture
{
protected:
    using sss_uniform_t = UniformAdjLogPDFNode<
        scl_expr_view_t, 
        scl_expr_view_t, 
        scl_expr_view_t>;

    using vss_uniform_t = UniformAdjLogPDFNode<
        vec_expr_view_t, 
        scl_expr_view_t, 
        scl_expr_view_t>;

    using vsv_uniform_t = UniformAdjLogPDFNode<
        vec_expr_view_t, 
        scl_expr_view_t, 
        vec_expr_view_t>;

    using vvs_uniform_t = UniformAdjLogPDFNode<
        vec_expr_view_t, 
        vec_expr_view_t, 
        scl_expr_view_t>;

    using vvv_uniform_t = UniformAdjLogPDFNode<
        vec_expr_view_t, 
        vec_expr_view_t, 
        vec_expr_view_t>;

    scl_expr_t scl_x;
    scl_expr_t scl_min;
    scl_expr_t scl_max;
    vec_expr_t vec_x;
    vec_expr_t vec_min;
    vec_expr_t vec_max;

    sss_uniform_t sss_uniform;
    vss_uniform_t vss_uniform;
    vsv_uniform_t vsv_uniform;
    vvs_uniform_t vvs_uniform;
    vvv_uniform_t vvv_uniform;

    value_t tol = 1e-15;

    uniform_fixture()
        : base_fixture()
        , vec_x(3)
        , vec_min(3)
        , vec_max(3)
        , sss_uniform(scl_x, scl_min, scl_max)
        , vss_uniform(vec_x, scl_min, scl_max)
        , vsv_uniform(vec_x, scl_min, vec_max)
        , vvs_uniform(vec_x, vec_min, scl_max)
        , vvv_uniform(vec_x, vec_min, vec_max)
    {
        // initialize some values
        scl_x.get() = 0.45;
        scl_min.get() = -3.2415;
        scl_max.get() = 0.5231;

        vec_x.get(0,0) = 0.5;
        vec_x.get(1,0) = -2.3;
        vec_x.get(2,0) = -3.2414999;

        vec_min.get(0,0) = 0.4;
        vec_min.get(1,0) = -2.30000001;
        vec_min.get(2,0) = -10.32;

        vec_max.get(0,0) = 0.51;
        vec_max.get(1,0) = 0.0;
        vec_max.get(2,0) = 3.4;
    }
};

TEST_F(uniform_fixture, sss_feval)
{
    bind(sss_uniform);
    value_t res = sss_uniform.feval();
    EXPECT_DOUBLE_EQ(res, -1.3256416139079406);
}

TEST_F(uniform_fixture, sss_beval)
{
    bind(sss_uniform);
    sss_uniform.feval();
    sss_uniform.beval(1, 0, 0, util::beval_policy::single);
    EXPECT_DOUBLE_EQ(scl_min.get_adj(0,0), 
                     0.2656324709132444);
    EXPECT_DOUBLE_EQ(scl_max.get_adj(0,0), 
                    -scl_min.get_adj(0,0));
}

TEST_F(uniform_fixture, vss_feval)
{
    bind(vss_uniform);
    value_t res = vss_uniform.feval();
    EXPECT_DOUBLE_EQ(res, -3.9769248417238217);
}

TEST_F(uniform_fixture, vss_beval)
{
    bind(vss_uniform);
    vss_uniform.feval();
    vss_uniform.beval(1, 0, 0, util::beval_policy::single);
    EXPECT_DOUBLE_EQ(scl_min.get_adj(0,0), 
                     0.7968974127397332);
    EXPECT_DOUBLE_EQ(scl_max.get_adj(0,0), 
                    -scl_min.get_adj(0,0));
}

TEST_F(uniform_fixture, vsv_feval)
{
    bind(vsv_uniform);
    vsv_uniform.feval();
    value_t res = vsv_uniform.feval();
    EXPECT_DOUBLE_EQ(res, -4.3915297872269807);
}

TEST_F(uniform_fixture, vsv_beval)
{
    bind(vsv_uniform);
    vsv_uniform.feval();
    vsv_uniform.beval(1, 0, 0, util::beval_policy::single);
    
    EXPECT_DOUBLE_EQ(scl_min.get_adj(0,0), 
                     0.7256275899706838);

    EXPECT_DOUBLE_EQ(vec_max.get_adj(0,0), 
                     -0.2665600426496068);
    EXPECT_DOUBLE_EQ(vec_max.get_adj(1,0), 
                     -0.3084991516273330);
    EXPECT_DOUBLE_EQ(vec_max.get_adj(2,0), 
                     -0.1505683956937439);
}

TEST_F(uniform_fixture, vvs_feval)
{
    bind(vvs_uniform);
    vvs_uniform.feval();
    value_t res = vvs_uniform.feval();
    EXPECT_DOUBLE_EQ(res, -1.3266062626903801);
}

TEST_F(uniform_fixture, vvs_beval)
{
    bind(vvs_uniform);
    vvs_uniform.feval();
    vvs_uniform.beval(1, 0, 0, util::beval_policy::single);
    
    EXPECT_DOUBLE_EQ(vec_min.get_adj(0,0), 
                     8.1234768480909842);
    EXPECT_DOUBLE_EQ(vec_min.get_adj(1,0), 
                     0.3542205364520543);
    EXPECT_DOUBLE_EQ(vec_min.get_adj(2,0), 
                     0.0922245483302746);

    EXPECT_DOUBLE_EQ(scl_max.get_adj(0,0), 
                     -8.5699219328733136);

}

TEST_F(uniform_fixture, vvv_feval)
{
    bind(vvv_uniform);
    vvv_uniform.feval();
    value_t res = vvv_uniform.feval();
    EXPECT_DOUBLE_EQ(res, -1.2444888363909485);
}

TEST_F(uniform_fixture, vvv_beval)
{
    bind(vvv_uniform);
    vvv_uniform.feval();
    vvv_uniform.beval(1, 0, 0, util::beval_policy::single);
    
    EXPECT_DOUBLE_EQ(vec_min.get_adj(0,0), 
                     9.0909090909090917);
    EXPECT_DOUBLE_EQ(vec_min.get_adj(1,0), 
                     0.4347826068052930);
    EXPECT_DOUBLE_EQ(vec_min.get_adj(2,0), 
                     0.0728862973760933);

    for (size_t i = 0; i < vec_max.size(); ++i) {
        EXPECT_DOUBLE_EQ(vec_max.get_adj(i,0), 
                         -vec_min.get_adj(i,0));
    }
}

} // namespace core
} // namespace ad
