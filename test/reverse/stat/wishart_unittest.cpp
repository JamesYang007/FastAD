#include <testutil/base_fixture.hpp>
#include <fastad_bits/reverse/stat/wishart.hpp>

namespace ad {
namespace stat {

struct wishart_fixture : base_fixture
{
protected:
    using const_t = core::Constant<size_t, scl>;
    using wishart_t = WishartAdjLogPDFNode<
        mat_expr_view_t, 
        mat_expr_view_t, 
        const_t>;

    Eigen::VectorXd x_flat_vals;
    Eigen::VectorXd x_flat_adjs;

    mat_expr_t x;
    mat_expr_t v;
    value_t n;

    wishart_t wishart;

    value_t tol = 1e-15;

    wishart_fixture()
        : base_fixture()
        , x_flat_vals(6)
        , x_flat_adjs(6)
        , x(3,3)
        , v(3,3)
        , n(4)
        , wishart(x, v, n)
    {
        // initialize some values
        x.get(0,0) = x_flat_vals(0) = 10;
        x.get(1,0) = x_flat_vals(1) = 2;
        x.get(2,0) = x_flat_vals(2) = 3;
        x.get(1,1) = x_flat_vals(3) = 10;
        x.get(2,1) = x_flat_vals(4) = 1;
        x.get(2,2) = x_flat_vals(5) = 10;

        // for regular matrices, should initialize them to be symmetric already
        // symmetric matrices should symmetrify already during evaluation
        x.get() = x.get().template selfadjointView<Eigen::Lower>();

        v.get(0,0) = 5;
        v.get(1,0) = 1;
        v.get(2,0) = 0;
        v.get(1,1) = 5;
        v.get(2,1) = 1;
        v.get(2,2) = 5;

        v.get() = v.get().template selfadjointView<Eigen::Lower>();

        // MUST initialize adjoints to 0
        x_flat_adjs.setZero();
    }
};

TEST_F(wishart_fixture, feval) 
{
    bind(wishart);
    value_t res = wishart.feval();
    EXPECT_DOUBLE_EQ(res, -12.55942947411780252764);
}

TEST_F(wishart_fixture, feval_invalid) 
{
    v.get().setZero();
    bind(wishart);
    value_t res = wishart.feval();
    EXPECT_DOUBLE_EQ(res, util::neg_inf<value_t>);
}

TEST_F(wishart_fixture, beval) 
{
    bind(wishart);
    wishart.feval();
    wishart.beval(1.);

    value_t p = v.rows();

    Eigen::MatrixXd v_inv = v.get().inverse();
    Eigen::MatrixXd dX = 0.5 * ((n-p-1) * x.get().inverse() - v_inv);
    Eigen::MatrixXd dV = 0.5 * ((v_inv * x.get() * v_inv) - n * v_inv);

    EXPECT_EQ(static_cast<size_t>(dX.rows()), x.rows());
    EXPECT_EQ(static_cast<size_t>(dX.cols()), x.cols());
    EXPECT_EQ(static_cast<size_t>(dV.rows()), v.rows());
    EXPECT_EQ(static_cast<size_t>(dV.cols()), v.cols());

    for (size_t i = 0; i < x.rows(); ++i) {
        for (size_t j = 0; j < x.cols(); ++j) {
            EXPECT_NEAR(x.get_adj(i,j), dX(i,j), tol);
        }
    }

    for (size_t i = 0; i < v.rows(); ++i) {
        for (size_t j = 0; j < v.cols(); ++j) {
            EXPECT_NEAR(v.get_adj(i,j), dV(i,j), tol);
        }
    }
}

} // namespace stat
} // namespace ad
