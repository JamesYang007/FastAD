#include <testutil/base_fixture.hpp>
#include <fastad_bits/reverse/core/dot.hpp>
#include <fastad_bits/reverse/core/unary.hpp>

namespace ad {
namespace core {

struct dot_fixture: base_fixture
{
protected:
    using unary_t = MockUnary;
    using vec_unary_t = UnaryNode<unary_t, vec_expr_view_t>;
    using mat_unary_t = UnaryNode<unary_t, mat_expr_view_t>;
    using dot_vars_t = DotNode<mat_expr_view_t, 
                               vec_expr_view_t>;
    using dot_unary_t = DotNode<mat_unary_t,
                                vec_unary_t>;
    using dot_mat_unary_t = DotNode<mat_unary_t,
                                    mat_unary_t>;

    mat_expr_t mat_expr_transpose;

    dot_vars_t dot_vars;
    dot_unary_t dot_unary;
    dot_mat_unary_t dot_mat_unary;

    aVectorXd vseed;
    Eigen::ArrayXXd mseed;

    dot_fixture()
        : base_fixture(3) // change vec_size to 3
        , mat_expr_transpose(3, 2)
        , dot_vars(mat_expr, vec_expr)
        , dot_unary({mat_expr}, {vec_expr})
        , dot_mat_unary({mat_expr}, {mat_expr_transpose})
        , vseed(2)
        , mseed(2, 2)
    {
        vseed << 2.3, 1.32;
        mseed << 2.34, -3.2, 3.4, 1.23455;

        // initialize vector since not default size
        vec_initialize();

        // initialize extra matrix variables
        mat_initialize();

        // resize to whatever is the max memory needed
        val_buf.resize(dot_vars.bind_cache_size()(0));
        val_buf.resize(dot_unary.bind_cache_size()(0));
        val_buf.resize(dot_mat_unary.bind_cache_size()(0));

        adj_buf.resize(dot_vars.bind_cache_size()(1));
        adj_buf.resize(dot_unary.bind_cache_size()(1));
        adj_buf.resize(dot_mat_unary.bind_cache_size()(1));

        // bind all expressions to buffer
        ptr_pack_t ptr_pack(val_buf.data(), adj_buf.data());
        dot_vars.bind_cache(ptr_pack);
        dot_unary.bind_cache(ptr_pack);
        dot_mat_unary.bind_cache(ptr_pack);
    }

    void vec_initialize() 
    {
        auto vec_raw = vec_expr.get();
        vec_raw(0) = 0.312332;
        vec_raw(1) = -3.213;
        vec_raw(2) = 9.135;
    }

    void mat_initialize()
    {
        auto mat_raw = mat_expr_transpose.get();
        mat_raw(0,0) = 2.3;
        mat_raw(1,0) = 0.523;
        mat_raw(2,0) = 4.2;
        mat_raw(0,1) = 0.;
        mat_raw(1,1) = -2.3;
        mat_raw(2,1) = 95.2;
    }
};

TEST_F(dot_fixture, dot_vars_feval)
{
    Eigen::VectorXd actual = dot_vars.feval();
    Eigen::VectorXd expected = mat_expr.get() * vec_expr.get();
    check_eq(actual, expected);
}

TEST_F(dot_fixture, dot_vars_beval)
{
    Eigen::MatrixXd ladj = vseed.matrix() * vec_expr.get().transpose();
    Eigen::VectorXd radj = mat_expr.get().transpose() * vseed.matrix();
    dot_vars.feval();
    dot_vars.beval(vseed);
    check_eq(ladj, mat_expr.get_adj());
    check_eq(radj, vec_expr.get_adj());
}

TEST_F(dot_fixture, dot_unary_feval)
{
    Eigen::VectorXd actual = dot_unary.feval();
    Eigen::VectorXd expected = 
        unary_t::fmap(mat_expr.get().array()).matrix() * 
        unary_t::fmap(vec_expr.get().array()).matrix();
    check_eq(actual, expected);
}

TEST_F(dot_fixture, dot_unary_beval)
{
    Eigen::MatrixXd umat = unary_t::fmap(mat_expr.get().array()).matrix();
    Eigen::VectorXd uvec = unary_t::fmap(vec_expr.get().array()).matrix();
    Eigen::MatrixXd ladj = unary_t::bmap(
            vseed.matrix() * uvec.transpose(), 
            mat_expr.get().array(), 
            umat);
    Eigen::VectorXd radj = unary_t::bmap(
            umat.transpose() * vseed.matrix(),
            vec_expr.get().array(),
            uvec);
    dot_unary.feval();
    dot_unary.beval(vseed);
    check_eq(ladj, mat_expr.get_adj());
    check_eq(radj, vec_expr.get_adj());
}

TEST_F(dot_fixture, dot_mat_unary_feval)
{
    Eigen::MatrixXd actual = dot_mat_unary.feval();
    Eigen::MatrixXd expected = 
        unary_t::fmap(mat_expr.get().array()).matrix() * 
        unary_t::fmap(mat_expr_transpose.get().array()).matrix();
    check_eq(actual, expected);
}

TEST_F(dot_fixture, dot_mat_unary_beval)
{
    Eigen::MatrixXd umat1 = unary_t::fmap(mat_expr.get().array()).matrix();
    Eigen::MatrixXd umat2 = unary_t::fmap(mat_expr_transpose.get().array()).matrix();
    Eigen::MatrixXd ladj = unary_t::bmap(
            mseed.matrix() * umat2.transpose(), 
            mat_expr.get().array(), 
            umat1);
    Eigen::MatrixXd radj = unary_t::bmap(
            umat1.transpose() * mseed.matrix(),
            mat_expr_transpose.get().array(),
            umat2);
    dot_mat_unary.feval();
    dot_mat_unary.beval(mseed);
    check_eq(ladj, mat_expr.get_adj());
    check_eq(radj, mat_expr_transpose.get_adj());
}

TEST_F(dot_fixture, dot_constant_var_feval)
{
    auto mat_const = ad::constant(mat_expr.get());
    auto dot_constant_var = ad::dot(mat_const, vec_expr);

    this->bind(dot_constant_var);

    Eigen::VectorXd actual = dot_constant_var.feval();
    Eigen::VectorXd expected = mat_expr.get() * vec_expr.get();
    check_eq(actual, expected);
}

TEST_F(dot_fixture, dot_constant_var_beval)
{
    auto mat_const = ad::constant(mat_expr.get());
    auto dot_constant_var = ad::dot(mat_const, vec_expr);

    this->bind(dot_constant_var);

    dot_constant_var.feval();
    dot_constant_var.beval(vseed);
    
    Eigen::VectorXd radj = mat_const.get().transpose() * vseed.matrix();
    check_eq(radj, vec_expr.get_adj());
}

TEST_F(dot_fixture, dot_constant)
{
    auto vec_const = ad::constant(vec_expr.get());
    auto dot_constant = ad::dot(mat_expr.get(), vec_const);

    static_assert(std::is_same_v<
            std::decay_t<decltype(dot_constant)>,
            Constant<value_t, ad::vec> >);

    Eigen::VectorXd actual = dot_constant.feval();
    Eigen::VectorXd expected = mat_expr.get() * vec_expr.get();
    for (int i = 0; i < expected.size(); ++i) {
        EXPECT_DOUBLE_EQ(actual(i), expected(i));
    }
}

} // namespace core
} // namespace ad
