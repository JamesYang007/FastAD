#include "gtest/gtest.h"
#include <fastad_bits/binary.hpp>
#include "base_fixture.hpp"

namespace ad {
namespace core {

struct binary_fixture : base_fixture
{
protected:
    using binary_t = MockBinary;
    using scl_scl_binary_t = 
        BinaryNode<binary_t, 
                   scl_expr_view_t, 
                   scl_expr_view_t>;
    using scl_scl_scl_binary_t = 
        BinaryNode<binary_t, 
                   scl_expr_view_t, 
                   scl_scl_binary_t>;
    using scl_vec_binary_t = 
        BinaryNode<binary_t, 
                   scl_expr_view_t,
                   vec_expr_view_t>;
    using vec_vec_binary_t = 
        BinaryNode<binary_t,
                   vec_expr_view_t,
                   vec_expr_view_t>;
    using mat_mat_binary_t = 
        BinaryNode<binary_t, 
                   mat_expr_view_t,
                   mat_expr_view_t>;

    scl_scl_binary_t scl_scl_binary;
    scl_vec_binary_t scl_vec_binary;
    vec_vec_binary_t vec_vec_binary;
    mat_mat_binary_t mat_mat_binary;
    scl_scl_scl_binary_t scl_scl_scl_binary;

    value_t seed = 3.14;

    std::vector<value_t> val_buf;

    binary_fixture()
        : base_fixture()
        , scl_scl_binary(scl_expr, scl_expr)
        , scl_vec_binary(scl_expr, vec_expr)
        , vec_vec_binary(vec_expr, vec_expr)
        , mat_mat_binary(mat_expr, mat_expr)
        , scl_scl_scl_binary(scl_expr, scl_scl_binary)
        , val_buf(std::max(vec_size, mat_size), 0)
    {
        // IMPORTANT: bind value for unary nodes.
        // No two unary node expressions can be used in a single test.
        scl_scl_binary.bind(val_buf.data());
        scl_vec_binary.bind(val_buf.data());
        vec_vec_binary.bind(val_buf.data());
        mat_mat_binary.bind(val_buf.data());
        scl_scl_scl_binary.bind(val_buf.data());
    }
};

TEST_F(binary_fixture, scl_scl_feval)
{
    value_t res = scl_scl_binary.feval();
    EXPECT_DOUBLE_EQ(res, -scl_expr.get());
}

TEST_F(binary_fixture, scl_scl_beval)
{
    scl_scl_binary.beval(seed, 0,0);    // last two ignored
    EXPECT_DOUBLE_EQ(scl_expr.get_adj(0,0), -seed);
}

TEST_F(binary_fixture, scl_vec_feval)
{
    Eigen::VectorXd res = scl_vec_binary.feval();
    for (int i = 0; i < res.size(); ++i) {
        value_t expected = scl_expr.get()-2*vec_expr.get(i,0); // last param ignored
        EXPECT_DOUBLE_EQ(res(i), expected);   
    }
}

TEST_F(binary_fixture, scl_vec_beval)
{
    scl_vec_binary.beval(seed,3,0);    // last param ignored
    scl_vec_binary.beval(seed,4,0);    // last param ignored
    EXPECT_DOUBLE_EQ(scl_expr.get_adj(0,0), 2*seed);
    for (size_t i = 0; i < vec_size; ++i) {
        if (i == 3 || i == 4) {
            EXPECT_DOUBLE_EQ(vec_expr.get_adj(i,0), -2*seed);
        } else {
            EXPECT_DOUBLE_EQ(vec_expr.get_adj(i,0), 0.);
        }
    }
}

TEST_F(binary_fixture, vec_vec_feval)
{
    Eigen::VectorXd res = vec_vec_binary.feval();
    for (int i = 0; i < res.size(); ++i) {
        value_t expected = -vec_expr.get(i,0); // last param ignored
        EXPECT_DOUBLE_EQ(res(i), expected);   
    }
}

TEST_F(binary_fixture, vec_vec_beval)
{
    vec_vec_binary.beval(seed, 0,0);    // last param ignored
    vec_vec_binary.beval(seed, 2,0);    // last param ignored
    for (size_t i = 0; i < vec_size; ++i) {
        value_t expected = (i == 0 || i == 2) ?
            -seed : 0.;
        EXPECT_DOUBLE_EQ(vec_expr.get_adj(i,0), expected);   
    }
}

TEST_F(binary_fixture, mat_mat_feval)
{
    Eigen::MatrixXd res = mat_mat_binary.feval();
    for (int i = 0; i < res.rows(); ++i) {
        for (int j = 0; j < res.cols(); ++j) {
            value_t expected = -mat_expr.get(i,j);
            EXPECT_DOUBLE_EQ(res(i,j), expected);   
        }
    }
}

TEST_F(binary_fixture, mat_mat_beval)
{
    mat_mat_binary.beval(seed, 0,1);
    mat_mat_binary.beval(seed, 1,2);
    for (size_t i = 0; i < mat_rows; ++i) {
        for (size_t j = 0; j < mat_cols; ++j) {
            value_t expected = ((i == 0 && j == 1) || 
                                (i == 1 && j == 2)) ?
                -seed : 0.;
            EXPECT_DOUBLE_EQ(mat_expr.get_adj(i,j), expected);   
        }
    }
}

TEST_F(binary_fixture, scl_scl_scl_feval)
{
    value_t res = scl_scl_scl_binary.feval();
    EXPECT_DOUBLE_EQ(res, 3.*scl_expr.get());
}

TEST_F(binary_fixture, scl_scl_scl_beval)
{
    scl_scl_scl_binary.beval(seed, 0,0);    // last two ignored
    EXPECT_DOUBLE_EQ(scl_expr.get_adj(0,0), 3.*seed);
}

} // namespace core
} // namespace ad
