#include <testutil/base_fixture.hpp>
#include <fastad_bits/reverse/stat/normal.hpp>

namespace ad {
namespace stat {

struct normal_fixture : base_fixture
{
protected:
    using sss_normal_t = NormalAdjLogPDFNode<
        scl_expr_view_t, 
        scl_expr_view_t, 
        scl_expr_view_t>;
    using vss_normal_t = NormalAdjLogPDFNode<
        vec_expr_view_t, 
        scl_expr_view_t, 
        scl_expr_view_t>;
    using vvs_normal_t = NormalAdjLogPDFNode<
        vec_expr_view_t, 
        vec_expr_view_t, 
        scl_expr_view_t>;
    using vsv_normal_t = NormalAdjLogPDFNode<
        vec_expr_view_t, 
        scl_expr_view_t, 
        vec_expr_view_t>;
    using vvv_normal_t = NormalAdjLogPDFNode<
        vec_expr_view_t, 
        vec_expr_view_t, 
        vec_expr_view_t>;
    using vsm_normal_t = NormalAdjLogPDFNode<
        vec_expr_view_t, 
        scl_expr_view_t, 
        mat_expr_view_t>;
    using vvm_normal_t = NormalAdjLogPDFNode<
        vec_expr_view_t, 
        vec_expr_view_t, 
        mat_expr_view_t>;

    scl_expr_t scl_x;
    scl_expr_t scl_mu;
    scl_expr_t scl_sigma;
    vec_expr_t vec_x;
    vec_expr_t vec_mu;
    vec_expr_t vec_sigma;
    mat_expr_t mat_sigma;

    sss_normal_t sss_normal;
    vss_normal_t vss_normal;
    vvs_normal_t vvs_normal;
    vsv_normal_t vsv_normal;
    vvv_normal_t vvv_normal;
    vsm_normal_t vsm_normal;
    vvm_normal_t vvm_normal;

    value_t tol = 1e-15;

    normal_fixture()
        : base_fixture()
        , vec_x(3)
        , vec_mu(3)
        , vec_sigma(3)
        , mat_sigma(3,3)
        , sss_normal(scl_x, scl_mu, scl_sigma)
        , vss_normal(vec_x, scl_mu, scl_sigma)
        , vvs_normal(vec_x, vec_mu, scl_sigma)
        , vsv_normal(vec_x, scl_mu, vec_sigma)
        , vvv_normal(vec_x, vec_mu, vec_sigma)
        , vsm_normal(vec_x, scl_mu, mat_sigma)
        , vvm_normal(vec_x, vec_mu, mat_sigma)
    {
        // initialize some values
        this->scl_initialize(scl_x);

        scl_mu.get() = -0.2;

        scl_sigma.get() = 0.01;

        vec_x.get(0,0) = 3.1;
        vec_x.get(1,0) = -2.3;
        vec_x.get(2,0) = 1.3;

        vec_mu.get(0,0) = -0.3;
        vec_mu.get(1,0) = -2.3;
        vec_mu.get(2,0) = -1.2;

        vec_sigma.get(0,0) = 0.01;
        vec_sigma.get(1,0) = 1.03;
        vec_sigma.get(2,0) = 2.41;

        mat_initialize(mat_sigma);
    }

    template <class ExprType>
    void mat_initialize(ExprType& expr)
    {
        expr.get(0,0) = 1.0;
        expr.get(1,0) = 0.3;
        expr.get(2,0) = 0.2;
        expr.get(1,1) = 2.0;
        expr.get(2,1) = -0.3;
        expr.get(2,2) = 3.0;
    }
};

TEST_F(normal_fixture, sss_feval)
{
    bind(sss_normal);
    value_t res = sss_normal.feval();
    EXPECT_DOUBLE_EQ(res, -31495.8948298140203406);
}

TEST_F(normal_fixture, sss_beval)
{
    bind(sss_normal);
    sss_normal.feval();
    sss_normal.beval(1.);

    EXPECT_DOUBLE_EQ(scl_x.get_adj(0,0), 
                     -25100.0000000000000000);
    EXPECT_DOUBLE_EQ(scl_mu.get_adj(0,0), 
                     25100.0000000000000000);
    EXPECT_DOUBLE_EQ(scl_sigma.get_adj(0,0), 
                     6300000.0000000009313226);
}

TEST_F(normal_fixture, vss_feval)
{
    bind(vss_normal);
    value_t res = vss_normal.feval();
    EXPECT_DOUBLE_EQ(res, -87736.1844894420355558);
}

TEST_F(normal_fixture, vss_beval)
{
    bind(vss_normal);
    vss_normal.feval();
    vss_normal.beval(1.);

    EXPECT_DOUBLE_EQ(vec_x.get_adj(0,0), 
                     -33000.0000000000000000);
    EXPECT_DOUBLE_EQ(vec_x.get_adj(1,0), 
                     20999.9999999999963620);
    EXPECT_DOUBLE_EQ(vec_x.get_adj(2,0), 
                     -15000.0000000000000000);
    EXPECT_DOUBLE_EQ(scl_mu.get_adj(0,0), 
                     27000.0000000000036380);
    EXPECT_DOUBLE_EQ(scl_sigma.get_adj(0,0), 
                     17549700.0000000000000000);
}

TEST_F(normal_fixture, vss_constant_feval)
{
    auto x = ad::constant(vec_x.get());
    auto vss_normal_constant = normal_adj_log_pdf(x, scl_mu, scl_sigma);
    bind(vss_normal_constant);
    value_t res = vss_normal_constant.feval();
    EXPECT_DOUBLE_EQ(res, -87736.1844894420355558);
}

TEST_F(normal_fixture, vss_constant_beval)
{
    auto x = ad::constant(vec_x.get());
    auto vss_normal_constant = normal_adj_log_pdf(x, scl_mu, scl_sigma);
    bind(vss_normal_constant);
    vss_normal_constant.feval();
    vss_normal_constant.beval(1.);

    EXPECT_DOUBLE_EQ(scl_mu.get_adj(0,0), 
                     27000.0000000000036380);
    EXPECT_DOUBLE_EQ(scl_sigma.get_adj(0,0), 
                     17549700.0000000000000000);
}

TEST_F(normal_fixture, vvs_feval)
{
    bind(vvs_normal);
    value_t res = vvs_normal.feval();
    EXPECT_DOUBLE_EQ(res, -89036.1844894420355558);
}

TEST_F(normal_fixture, vvs_beval)
{
    bind(vvs_normal);
    vvs_normal.feval();
    vvs_normal.beval(1.);

    EXPECT_DOUBLE_EQ(vec_x.get_adj(0,0), 
                     -34000.0000000000000000);
    EXPECT_DOUBLE_EQ(vec_x.get_adj(1,0), 
                     0.0);
    EXPECT_DOUBLE_EQ(vec_x.get_adj(2,0), 
                     -25000.0000000000000000);
    for (size_t i = 0; i < vec_x.rows(); ++i) {
        EXPECT_DOUBLE_EQ(vec_mu.get_adj(i,0), -vec_x.get_adj(i,0));
    }
    EXPECT_DOUBLE_EQ(scl_sigma.get_adj(0,0), 
                     17809700.0000000000000000);
}

TEST_F(normal_fixture, vsv_feval)
{
    bind(vsv_normal);
    value_t res = vsv_normal.feval();
    EXPECT_DOUBLE_EQ(res, -54448.5761343555350322);
}

TEST_F(normal_fixture, vsv_feval_not_pos_def)
{
    vec_sigma.get()(0) = -1;
    bind(vsv_normal);
    value_t res = vsv_normal.feval();
    EXPECT_DOUBLE_EQ(res, util::neg_inf<value_t>);
}

TEST_F(normal_fixture, vsv_beval)
{
    bind(vsv_normal);
    vsv_normal.feval();
    vsv_normal.beval(1.);
    
    EXPECT_DOUBLE_EQ(vec_x.get_adj(0,0), 
                     -33000.0000000000000000);
    EXPECT_DOUBLE_EQ(vec_x.get_adj(1,0), 
                     1.9794514091808839);
    EXPECT_DOUBLE_EQ(vec_x.get_adj(2,0), 
                     -0.2582600161842943);
    EXPECT_DOUBLE_EQ(scl_mu.get_adj(0,0), 
                     32998.2788086070067948);
    EXPECT_DOUBLE_EQ(vec_sigma.get_adj(0,0), 
                     10889900.0000000000000000);
    EXPECT_DOUBLE_EQ(vec_sigma.get_adj(1,0), 
                     3.0649009313396647);
    EXPECT_DOUBLE_EQ(vec_sigma.get_adj(2,0), 
                     -0.2541950106736757);
}

TEST_F(normal_fixture, vsv_constant_feval)
{
    auto x = ad::constant(vec_x.get());
    auto s = ad::constant(vec_sigma.get());
    auto vsv_normal_constant = normal_adj_log_pdf(x, scl_mu, s);
    bind(vsv_normal_constant);
    value_t res = vsv_normal_constant.feval();
    EXPECT_DOUBLE_EQ(res, -54448.5761343555350322);
}

TEST_F(normal_fixture, vsv_constant_beval)
{
    auto x = ad::constant(vec_x.get());
    auto s = ad::constant(vec_sigma.get());
    auto vsv_normal_constant = normal_adj_log_pdf(x, scl_mu, s);
    bind(vsv_normal_constant);
    vsv_normal_constant.feval();
    vsv_normal_constant.beval(1.);
    EXPECT_DOUBLE_EQ(scl_mu.get_adj(0,0), 
                     32998.2788086070067948);
}

TEST_F(normal_fixture, vvv_feval)
{
    bind(vvv_normal);
    value_t res = vvv_normal.feval();
    EXPECT_DOUBLE_EQ(res, -57796.8420570641465019);
}

TEST_F(normal_fixture, vvv_feval_not_pos_def)
{
    vec_sigma.get(0,0) = 0;
    bind(vvv_normal);
    value_t res = vvv_normal.feval();
    EXPECT_DOUBLE_EQ(res, util::neg_inf<value_t>);
}

TEST_F(normal_fixture, vvv_beval)
{
    bind(vvv_normal);
    vvv_normal.feval();
    vvv_normal.beval(1.);

    EXPECT_DOUBLE_EQ(vec_x.get_adj(0,0), 
                     -34000.0000000000000000);
    EXPECT_DOUBLE_EQ(vec_x.get_adj(1,0), 
                     0.0);
    EXPECT_DOUBLE_EQ(vec_x.get_adj(2,0), 
                     -0.4304333603071572);

    for (size_t i = 0; i < vec_x.rows(); ++i) {
        EXPECT_DOUBLE_EQ(vec_mu.get_adj(i,0), -vec_x.get_adj(i,0));
    }

    EXPECT_DOUBLE_EQ(vec_sigma.get_adj(0,0), 
                     11559900.0000000000000000);
    EXPECT_DOUBLE_EQ(vec_sigma.get_adj(1,0), 
                     -0.9708737864077670);
    EXPECT_DOUBLE_EQ(vec_sigma.get_adj(2,0), 
                     0.0315698758372999);
}

TEST_F(normal_fixture, vsm_feval)
{
    bind(vsm_normal);
    value_t res = vsm_normal.feval();
    EXPECT_DOUBLE_EQ(res, -8.8105250497069019);
}

TEST_F(normal_fixture, vsm_feval_not_pos_def)
{
    mat_sigma.get().setZero();
    bind(vsm_normal);
    value_t res = vsm_normal.feval();
    EXPECT_DOUBLE_EQ(res, util::neg_inf<value_t>);
}

TEST_F(normal_fixture, vsm_beval)
{
    bind(vsm_normal);
    vsm_normal.feval();
    vsm_normal.beval(1.);

    EXPECT_DOUBLE_EQ(vec_x.get_adj(0,0), 
                     -3.7624909485879803);
    EXPECT_DOUBLE_EQ(vec_x.get_adj(1,0), 
                     1.6010137581462704);
    EXPECT_DOUBLE_EQ(vec_x.get_adj(2,0), 
                     -0.0890658942795076);

    EXPECT_DOUBLE_EQ(scl_mu.get_adj(0,0), 2.2505430847212176);

    EXPECT_DOUBLE_EQ(mat_sigma.get_adj(0,0), 
                     6.5432306187049774);
    EXPECT_DOUBLE_EQ(mat_sigma.get_adj(1,0), 
                     -2.9250063314004424);
    EXPECT_NEAR(mat_sigma.get_adj(2,0), 
                0.2119067294266190,
                tol);
    EXPECT_DOUBLE_EQ(mat_sigma.get_adj(1,1),
                1.0137007310866775);
    EXPECT_DOUBLE_EQ(mat_sigma.get_adj(2,1),
                -0.1038829443345370);
    EXPECT_DOUBLE_EQ(mat_sigma.get_adj(2,2),
                -0.1689156028253514);
}

TEST_F(normal_fixture, vvm_feval)
{
    bind(vvm_normal);
    value_t res = vvm_normal.feval();
    EXPECT_DOUBLE_EQ(res, -7.3649692930088602);
}

TEST_F(normal_fixture, vvm_feval_not_pos_def)
{
    mat_sigma.get().setZero();
    bind(vvm_normal);
    value_t res = vvm_normal.feval();
    EXPECT_DOUBLE_EQ(res, util::neg_inf<value_t>);
}

TEST_F(normal_fixture, vvm_beval)
{
    bind(vvm_normal);
    vvm_normal.feval();
    vvm_normal.beval(1.);

    EXPECT_DOUBLE_EQ(vec_x.get_adj(0,0), 
                     -3.4158218682114407);
    EXPECT_DOUBLE_EQ(vec_x.get_adj(1,0), 
                     0.4279507603186097);
    EXPECT_DOUBLE_EQ(vec_x.get_adj(2,0), 
                     -0.5628167994207096);

    for (size_t i = 0; i < vec_mu.rows(); ++i) {
        EXPECT_DOUBLE_EQ(vec_mu.get_adj(i,0), -vec_x.get_adj(i,0));
    }

    EXPECT_DOUBLE_EQ(mat_sigma.get_adj(0,0), 
                     5.2989810672774862);
    EXPECT_DOUBLE_EQ(mat_sigma.get_adj(1,0), 
                     -0.6440082274123684);
    EXPECT_DOUBLE_EQ(mat_sigma.get_adj(2,0), 
                    1.0055928845283644);
    EXPECT_DOUBLE_EQ(mat_sigma.get_adj(0,1), 
                     -0.6440082274123684);
    EXPECT_DOUBLE_EQ(mat_sigma.get_adj(1,1),
                    -0.1763508691715067);
    EXPECT_DOUBLE_EQ(mat_sigma.get_adj(2,1),
                    -0.1530140218890801);
    EXPECT_DOUBLE_EQ(mat_sigma.get_adj(0,2), 
                    1.0055928845283644);
    EXPECT_DOUBLE_EQ(mat_sigma.get_adj(1,2),
                    -0.1530140218890801);
    EXPECT_NEAR(mat_sigma.get_adj(2,2),
                -0.0145005947321700,
                tol);
}

} // namespace stat
} // namespace ad
