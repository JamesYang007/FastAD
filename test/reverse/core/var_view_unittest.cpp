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
    using selfadjmat_view_t = VarView<value_t, selfadjmat>;

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
    selfadjmat_view_t selfadj_matrix;

    var_view_fixture()
        : scalar(val_buf.data(), adj_buf.data())
        , vector(val_buf.data(), adj_buf.data(), vector_size)
        , matrix(val_buf.data(), adj_buf.data(),
                 matrix_rows, matrix_cols)
        , selfadj_matrix(val_buf.data(), adj_buf.data(),
                         matrix_rows, matrix_rows)
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
    scalar.beval(3.,0,0, util::beval_policy::single); // last two params ignored
    EXPECT_DOUBLE_EQ(adj_buf[0], 3.);
}

TEST_F(var_view_fixture, scl_bind)
{
    auto next = scalar.bind(val_buf.data() + 1);
    EXPECT_EQ(next, val_buf.data() + 2);
    EXPECT_DOUBLE_EQ(scalar.feval(), val_buf[1]);
}

TEST_F(var_view_fixture, scl_bind_adj)
{
    auto next = scalar.bind_adj(adj_buf.data() + 1);
    EXPECT_EQ(next, adj_buf.data() + 2);
    scalar.beval(5.,0,0, util::beval_policy::single);   // last two params ignored
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

TEST_F(var_view_fixture, vec_beval)
{
    vector.beval(3., 1, 0, util::beval_policy::single); // last param ignored
    std::vector<value_t> actual(vector_size, 0.);
    actual[1] = 3.;
    compare_vectors(adj_buf, actual);
}

TEST_F(var_view_fixture, vec_bind)
{
    size_t offset = 1;
    auto next = vector.bind(val_buf.data() + offset);
    EXPECT_EQ(next, val_buf.data() + vector_size + offset);
    std::vector<value_t> actual(val_buf.data()+offset,
                                val_buf.data()+offset+vector_size);
    compare_vectors(vector.feval(), actual);
}

TEST_F(var_view_fixture, vec_bind_adj)
{
    size_t offset = 1;
    auto next = vector.bind_adj(adj_buf.data() + offset);
    EXPECT_EQ(next, adj_buf.data() + offset + vector_size);
    vector.beval(5., 2, 0, util::beval_policy::single); // last param ignored
    std::vector<value_t> res(adj_buf.data()+offset,
                             adj_buf.data()+offset+vector_size);
    std::vector<value_t> actual(vector_size, 0.);
    actual[2] = 5.;
    compare_vectors(res, actual);
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

TEST_F(var_view_fixture, mat_beval)
{
    matrix.beval(3., 1, 2, util::beval_policy::single);
    std::vector<value_t> actual(matrix_size, 0.);
    actual[1 + 2*matrix_rows] = 3.;
    compare_vectors(adj_buf, actual);
}

TEST_F(var_view_fixture, mat_bind)
{
    size_t offset = 1;
    auto next = matrix.bind(val_buf.data() + offset);
    EXPECT_EQ(next, val_buf.data() + matrix_size + offset);
    std::vector<value_t> actual(val_buf.data()+offset,
                                val_buf.data()+offset+matrix_size);
    compare_matrix_vector(matrix.feval(), actual);
}

TEST_F(var_view_fixture, mat_bind_adj)
{
    size_t offset = 1;
    auto next = matrix.bind_adj(adj_buf.data() + offset);
    EXPECT_EQ(next, adj_buf.data() + offset + matrix_size);
    matrix.beval(5., 0, 2, util::beval_policy::single);
    std::vector<value_t> res(adj_buf.data()+offset,
                             adj_buf.data()+offset+matrix_size);
    std::vector<value_t> actual(matrix_size, 0.);
    actual[2*matrix_rows] = 5.;
    compare_vectors(res, actual);
}

// TEST selfadjoint matrix

TEST_F(var_view_fixture, selfadjmat_rows)
{
    EXPECT_EQ(selfadj_matrix.rows(), matrix_rows);
}

TEST_F(var_view_fixture, selfadjmat_cols)
{
    EXPECT_EQ(selfadj_matrix.cols(), matrix_rows);
}

TEST_F(var_view_fixture, selfadjmat_size)
{
    EXPECT_EQ(selfadj_matrix.size(), 
              matrix_rows * matrix_rows);
}

TEST_F(var_view_fixture, selfadjmat_feval)
{
    auto res = selfadj_matrix.feval();
    for (int j = 0; j < res.cols(); ++j) {
        for (int i = 0; i < j; ++i) {
            EXPECT_DOUBLE_EQ(res(i,j),
                             val_buf[j + selfadj_matrix.rows() * i]);
        }
        for (int i = j; i < res.rows(); ++i) {
            EXPECT_DOUBLE_EQ(res(i,j),
                             val_buf[i + selfadj_matrix.rows() * j]);
        }
    }
}

TEST_F(var_view_fixture, selfadjmat_beval)
{
    selfadj_matrix.beval(3., 0, 1, util::beval_policy::single);
    std::vector<value_t> actual(selfadj_matrix.size(), 0.);
    actual[1] = 3.; // updates the lower half
    compare_vectors(adj_buf, actual);
}

TEST_F(var_view_fixture, selfadjmat_bind)
{
    size_t offset = 1;
    auto next = selfadj_matrix.bind(val_buf.data() + offset);
    EXPECT_EQ(next, val_buf.data() + selfadj_matrix.size() + offset);
    std::vector<value_t> actual(val_buf.data()+offset,
                                val_buf.data()+offset+selfadj_matrix.size());
    auto res = selfadj_matrix.feval();
    for (int j = 0; j < res.cols(); ++j) {
        for (int i = 0; i < j; ++i) {
            EXPECT_DOUBLE_EQ(res(i,j),
                             actual[j + selfadj_matrix.rows() * i]);
        }
        for (int i = j; i < res.rows(); ++i) {
            EXPECT_DOUBLE_EQ(res(i,j),
                             actual[i + selfadj_matrix.rows() * j]);
        }
    }
}

TEST_F(var_view_fixture, selfadjmat_bind_adj)
{
    size_t offset = 1;
    auto next = selfadj_matrix.bind_adj(adj_buf.data() + offset);
    EXPECT_EQ(next, adj_buf.data() + offset + selfadj_matrix.size());
    selfadj_matrix.beval(5., 0, 0, util::beval_policy::single);
    selfadj_matrix.beval(5., 1, 0, util::beval_policy::single); // updates lower half
    selfadj_matrix.beval(5., 0, 1, util::beval_policy::single); // updates lower half
    std::vector<value_t> res(adj_buf.data()+offset,
                             adj_buf.data()+offset+selfadj_matrix.size());
    std::vector<value_t> actual(selfadj_matrix.size(), 0.);
    actual[0] = 5.;
    actual[1] = 10.;
    compare_vectors(res, actual);
}

} // namespace ad
