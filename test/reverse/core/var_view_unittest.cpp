#include "gtest/gtest.h"
#include <array>
#include <fastad_bits/reverse/core/var_view.hpp>

namespace ad {

struct var_view_fixture : ::testing::Test
{
protected:
    using value_t = double;
    using scl_view_t = VarView<value_t, scl>;
    using vec_view_t = VarView<value_t, vec>;
    using mat_view_t = VarView<value_t, mat>;

    static constexpr size_t max_size = 10;
    static constexpr size_t vector_size = 5;
    static constexpr size_t matrix_rows = 2;
    static constexpr size_t matrix_cols = 3;
    static constexpr size_t matrix_size = matrix_rows*matrix_cols;

    std::array<value_t, max_size> val_buf = {0};
    std::array<value_t, max_size> adj_buf = {0};

    scl_view_t scalar;
    vec_view_t vector;
    mat_view_t matrix;

    var_view_fixture()
        : scalar(val_buf.data(), adj_buf.data())
        , vector(val_buf.data(), adj_buf.data(), vector_size)
        , matrix(val_buf.data(), adj_buf.data(),
                 matrix_rows, matrix_cols)
    {
        // initialize value buffer with some random (fixed)
        // unique numbers for testing purposes
        val_buf[0] = -4.1;
        val_buf[1] = -1;
        val_buf[2] = 3;
        val_buf[3] = 6.;
        val_buf[4] = 0.3;
        val_buf[5] = 2.4;
        val_buf[6] = 6;
        val_buf[7] = 3.2;
        val_buf[8] = -9.1;
        val_buf[9] = -10.98;
    }

    template <class V1, class V2>
    void compare_vectors(const V1& v1,
                         const V2& v2)
    {
        size_t min_size = std::min(
                static_cast<size_t>(v1.size()),
                static_cast<size_t>(v2.size())
                );
        for (size_t i = 0; i < min_size; ++i) {
            if constexpr(std::is_invocable_v<V1, size_t>) {
                EXPECT_DOUBLE_EQ(v1(i), v2[i]);
            } else {
                EXPECT_DOUBLE_EQ(v1[i], v2[i]);
            }
        }
    }

    // Assumes column-major matrix
    template <class M, class V>
    void compare_matrix_vector(const M& m,
                               const V& v)
    {
        for (int i = 0; i < m.rows(); ++i) {
            for (int j = 0; j < m.cols(); ++j) {
                if constexpr(std::is_invocable_v<V, size_t>) {
                    EXPECT_DOUBLE_EQ(m(i,j), v(i+m.rows()*j));
                } else {
                    EXPECT_DOUBLE_EQ(m(i,j), v[i+m.rows()*j]);
                }
            }
        }
    }
};

// TEST scalar

TEST_F(var_view_fixture, scl_feval)
{
    EXPECT_DOUBLE_EQ(scalar.feval(), val_buf[0]);
}

TEST_F(var_view_fixture, scl_beval)
{
    scalar.beval(3.);
    EXPECT_DOUBLE_EQ(adj_buf[0], 3.);
}

TEST_F(var_view_fixture, scl_bind)
{
    auto next = scalar.bind({val_buf.data() + 1, nullptr});
    EXPECT_EQ(next.val, val_buf.data() + 2);
    EXPECT_DOUBLE_EQ(scalar.feval(), val_buf[1]);
}

TEST_F(var_view_fixture, scl_bind_adj)
{
    auto next = scalar.bind({nullptr, adj_buf.data() + 1});
    EXPECT_EQ(next.adj, adj_buf.data() + 2);
    scalar.beval(5.); 
    EXPECT_DOUBLE_EQ(5., adj_buf[1]);
}

TEST_F(var_view_fixture, scl_size)
{
    EXPECT_EQ(scalar.size(), 1ul);
}

// TEST vector

TEST_F(var_view_fixture, vec_size)
{
    EXPECT_EQ(vector.size(), vector_size);
}

TEST_F(var_view_fixture, vec_feval)
{
    compare_vectors(vector.feval(), val_buf);
}

TEST_F(var_view_fixture, vec_beval_with_scl)
{
    vector.beval(3.);
    Eigen::VectorXd actual(vector_size);
    actual.array() = 3.;
    compare_vectors(adj_buf, actual);
}

TEST_F(var_view_fixture, vec_beval_with_vec)
{
    Eigen::VectorXd seed(vector_size);
    seed.setZero();
    seed(1) = 3;
    vector.beval(seed.array());
    compare_vectors(adj_buf, seed);
}

TEST_F(var_view_fixture, vec_bind)
{
    size_t offset = 1;
    auto next = vector.bind({val_buf.data() + offset, nullptr});
    EXPECT_EQ(next.val, val_buf.data() + vector_size + offset);
    std::vector<value_t> actual(val_buf.data()+offset,
                                val_buf.data()+offset+vector_size);
    compare_vectors(vector.feval(), actual);
}

TEST_F(var_view_fixture, vec_bind_adj)
{
    size_t offset = 1;
    auto next = vector.bind({nullptr, adj_buf.data() + offset});
    EXPECT_EQ(next.adj, adj_buf.data() + offset + vector_size);
    vector.beval(5);
    std::vector<value_t> res(adj_buf.data()+offset,
                             adj_buf.data()+offset+vector_size);
    Eigen::VectorXd actual(vector_size);
    actual.array() = 5.;
    compare_vectors(res, actual);
}

TEST_F(var_view_fixture, vec_subscript)
{
    vector.bind({val_buf.data(), nullptr});
    auto s = vector[1];
    EXPECT_EQ(s.size(), 1ul);
    EXPECT_EQ(s.data(), val_buf.data() + 1);
}

TEST_F(var_view_fixture, vec_call_operator)
{
    vector.bind({val_buf.data(), nullptr});
    auto s = vector(2);
    EXPECT_EQ(s.size(), 1ul);
    EXPECT_EQ(s.data(), val_buf.data() + 2);
}

// TEST matrix

TEST_F(var_view_fixture, mat_rows)
{
    EXPECT_EQ(matrix.rows(), matrix_rows);
}

TEST_F(var_view_fixture, mat_cols)
{
    EXPECT_EQ(matrix.cols(), matrix_cols);
}

TEST_F(var_view_fixture, mat_size)
{
    EXPECT_EQ(matrix.size(), matrix_size);
}

TEST_F(var_view_fixture, mat_feval)
{
    compare_matrix_vector(matrix.feval(), val_buf);
}

TEST_F(var_view_fixture, mat_beval_scl)
{
    matrix.beval(3.);
    Eigen::VectorXd actual(matrix_size);
    actual.array() = 3.;
    compare_vectors(adj_buf, actual);
}

TEST_F(var_view_fixture, mat_beval_mat)
{
    Eigen::MatrixXd seed(matrix_rows, matrix_cols);
    seed.setZero();
    seed(1,2) = 3.;
    matrix.beval(seed.array());
    std::vector<value_t> actual(matrix_size, 0.);
    actual[1 + 2*matrix_rows] = 3.;
    compare_vectors(adj_buf, actual);
}

TEST_F(var_view_fixture, mat_bind)
{
    size_t offset = 1;
    auto next = matrix.bind({val_buf.data() + offset, nullptr});
    EXPECT_EQ(next.val, val_buf.data() + matrix_size + offset);
    std::vector<value_t> actual(val_buf.data()+offset,
                                val_buf.data()+offset+matrix_size);
    compare_matrix_vector(matrix.feval(), actual);
}

TEST_F(var_view_fixture, mat_bind_adj)
{
    size_t offset = 1;
    auto next = matrix.bind({nullptr, adj_buf.data() + offset});
    EXPECT_EQ(next.adj, adj_buf.data() + offset + matrix_size);
    matrix.beval(5.);
    std::vector<value_t> res(adj_buf.data()+offset,
                             adj_buf.data()+offset+matrix_size);
    Eigen::VectorXd actual(matrix_size);
    actual.array() = 5.;
    compare_vectors(res, actual);
}

} // namespace ad
