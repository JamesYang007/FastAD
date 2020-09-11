#include <testutil/base_fixture.hpp>
#include <fastad_bits/reverse/core/constant.hpp>
#include <fastad_bits/reverse/core/det.hpp>

namespace ad {
namespace core {

struct det_fixture : base_fixture
{
protected:
    using det_llt_t = DetNode<DetLLT<value_t>, mat_expr_view_t>;
    using det_ldlt_t = DetNode<DetLDLT<value_t>, mat_expr_view_t>;
    using det_fplu_t = DetNode<DetFullPivLU<value_t>, mat_expr_view_t>;

    value_t seed = 9.2313;
    det_llt_t det_llt;
    det_ldlt_t det_ldlt;
    det_fplu_t det_fplu;

    det_fixture()
        : base_fixture(5, 4, 4) 
        , det_llt{mat_expr}
        , det_ldlt{mat_expr}
        , det_fplu{mat_expr}
    {
        // note that all these expressions require the same
        // cache sizes, so we may bind all of them to the same cache at once
        // so long as only one expression gets evaluated in each test
        this->bind(det_llt);
        this->bind(det_ldlt);
        this->bind(det_fplu);
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
TEST_F(det_fixture, det_fplu_feval)
{
    init_fplu();
    value_t actual = mat_expr.get().determinant();
    value_t res = det_fplu.feval();
    EXPECT_DOUBLE_EQ(res, actual);
}

TEST_F(det_fixture, det_fplu_beval)
{
    init_fplu();
    Eigen::MatrixXd adj = 
        seed * mat_expr.get().determinant() *
        mat_expr.get().inverse().transpose();

    det_fplu.feval();
    det_fplu.beval(seed);
    check_near(mat_expr.get_adj(), adj, 3e-13);
}

TEST_F(det_fixture, const_det_fplu_feval)
{
    init_fplu();
    auto mat = ad::constant(mat_expr.get());
    value_t actual = mat.get().determinant();
    auto res = ad::det(mat);
    static_assert(std::is_same_v<
            std::decay_t<decltype(res)>,
            Constant<value_t, scl>
            >);
    EXPECT_DOUBLE_EQ(res.get(), actual);
}

// LDLT
TEST_F(det_fixture, det_ldlt_feval)
{
    init_ldlt();
    value_t actual = mat_expr.get().determinant();
    value_t res = det_ldlt.feval();
    EXPECT_DOUBLE_EQ(res, actual);
}

TEST_F(det_fixture, det_ldlt_beval)
{
    init_ldlt();
    Eigen::MatrixXd adj = 
        seed * mat_expr.get().determinant() *
        mat_expr.get().inverse().transpose();

    det_ldlt.feval();
    det_ldlt.beval(seed);
    check_near(mat_expr.get_adj(), adj, 3e-13);
}

TEST_F(det_fixture, const_det_ldlt_feval)
{
    init_ldlt();
    auto mat = ad::constant(mat_expr.get());
    value_t actual = mat.get().determinant();
    auto res = ad::det<DetLDLT>(mat);
    static_assert(std::is_same_v<
            std::decay_t<decltype(res)>,
            Constant<value_t, scl>
            >);
    EXPECT_DOUBLE_EQ(res.get(), actual);
}

// LLT
TEST_F(det_fixture, det_llt_feval)
{
    init_llt();
    value_t actual = mat_expr.get().determinant();
    value_t res = det_llt.feval();
    EXPECT_DOUBLE_EQ(res, actual);
}

TEST_F(det_fixture, det_llt_beval)
{
    init_llt();
    Eigen::MatrixXd adj = 
        seed * mat_expr.get().determinant() *
        mat_expr.get().inverse().transpose();

    det_llt.feval();
    det_llt.beval(seed);
    check_near(mat_expr.get_adj(), adj, 3e-13);
}

TEST_F(det_fixture, const_det_llt_feval)
{
    init_llt();
    auto mat = ad::constant(mat_expr.get());
    value_t actual = mat.get().determinant();
    auto res = ad::det<DetLLT>(mat);
    static_assert(std::is_same_v<
            std::decay_t<decltype(res)>,
            Constant<value_t, scl>
            >);
    EXPECT_DOUBLE_EQ(res.get(), actual);
}

} // namespace core
} // namespace ad
