#include <testutil/base_fixture.hpp>
#include <fastad_bits/reverse/core/constant.hpp>
#include <fastad_bits/reverse/core/log_det.hpp>

namespace ad {
namespace core {

struct log_det_fixture : base_fixture
{
protected:
    using log_det_llt_t = LogDetNode<LogDetLLT<value_t>, mat_expr_view_t>;
    using log_det_ldlt_t = LogDetNode<LogDetLDLT<value_t>, mat_expr_view_t>;
    using log_det_fplu_t = LogDetNode<LogDetFullPivLU<value_t>, mat_expr_view_t>;

    value_t seed = 9.2313;
    log_det_llt_t log_det_llt;
    log_det_ldlt_t log_det_ldlt;
    log_det_fplu_t log_det_fplu;

    log_det_fixture()
        : base_fixture(5, 4, 4) 
        , log_det_llt{mat_expr}
        , log_det_ldlt{mat_expr}
        , log_det_fplu{mat_expr}
    {
        // note that all these expressions require the same
        // cache sizes, so we may bind all of them to the same cache at once
        // so long as only one expression gets evaluated in each test
        this->bind(log_det_llt);
        this->bind(log_det_ldlt);
        this->bind(log_det_fplu);
    }

    void init_fplu()
    {
        mat_expr.get() << 2, 3, 1, 5,
                          3, 5, -1, 3,
                          -2, 3, 1, 0,
                          -1, -1, 2, 7; 
    }

    void init_ldlt() 
    {
        init_llt();
    }

    void init_llt()
    {
        mat_expr.get() << 8, 3, 1, 5,
                          3, 5, 1, 3,
                          1, 1, 1, 0,
                          5, 3, 0, 7; 
    }
};

// FullPivLU
TEST_F(log_det_fixture, log_det_fplu_feval)
{
    init_fplu();
    value_t actual = std::log(std::abs(mat_expr.get().determinant()));
    value_t res = log_det_fplu.feval();
    EXPECT_DOUBLE_EQ(res, actual);
}

TEST_F(log_det_fixture, log_det_fplu_beval)
{
    init_fplu();
    Eigen::MatrixXd adj = 
        seed * mat_expr.get().inverse().transpose();

    log_det_fplu.feval();
    log_det_fplu.beval(seed);
    check_near(mat_expr.get_adj(), adj, 3e-13);
}

TEST_F(log_det_fixture, const_log_det_fplu_feval)
{
    init_fplu();
    auto mat = ad::constant(mat_expr.get());
    value_t actual = std::log(std::abs(mat.get().determinant()));
    auto res = ad::log_det(mat);
    static_assert(std::is_same_v<
            std::decay_t<decltype(res)>,
            Constant<value_t, scl>
            >);
    EXPECT_DOUBLE_EQ(res.get(), actual);
}

// LDLT
TEST_F(log_det_fixture, log_det_ldlt_feval)
{
    init_ldlt();
    value_t actual = std::log(std::abs(mat_expr.get().determinant()));
    value_t res = log_det_ldlt.feval();
    EXPECT_DOUBLE_EQ(res, actual);
}

TEST_F(log_det_fixture, log_det_ldlt_beval)
{
    init_ldlt();
    Eigen::MatrixXd adj = 
        seed * mat_expr.get().inverse().transpose();

    log_det_ldlt.feval();
    log_det_ldlt.beval(seed);
    check_near(mat_expr.get_adj(), adj, 3e-13);
}

TEST_F(log_det_fixture, const_log_det_ldlt_feval)
{
    init_ldlt();
    auto mat = ad::constant(mat_expr.get());
    value_t actual = std::log(std::abs(mat.get().determinant()));
    auto res = ad::log_det<LogDetLDLT>(mat);
    static_assert(std::is_same_v<
            std::decay_t<decltype(res)>,
            Constant<value_t, scl>
            >);
    EXPECT_DOUBLE_EQ(res.get(), actual);
}

// LLT
TEST_F(log_det_fixture, log_det_llt_feval)
{
    init_llt();
    value_t actual = std::log(std::abs(mat_expr.get().determinant()));
    value_t res = log_det_llt.feval();
    EXPECT_DOUBLE_EQ(res, actual);
}

TEST_F(log_det_fixture, log_det_llt_beval)
{
    init_llt();
    Eigen::MatrixXd adj = 
        seed * mat_expr.get().inverse().transpose();

    log_det_llt.feval();
    log_det_llt.beval(seed);
    check_near(mat_expr.get_adj(), adj, 3e-13);
}

TEST_F(log_det_fixture, const_log_det_llt_feval)
{
    init_llt();
    auto mat = ad::constant(mat_expr.get());
    value_t actual = std::log(std::abs(mat.get().determinant()));
    auto res = ad::log_det<LogDetLLT>(mat);
    static_assert(std::is_same_v<
            std::decay_t<decltype(res)>,
            Constant<value_t, scl>
            >);
    EXPECT_DOUBLE_EQ(res.get(), actual);
}

} // namespace core
} // namespace ad
