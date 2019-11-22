#include <fastad_bits/adeval.hpp>
#include <fastad_bits/admath.hpp>
#include <fastad_bits/adjacobian.hpp>
#include "gtest/gtest.h"
#include <random>
#include <time.h>
#include <armadillo>

namespace {

// TEST for Function Object
//
// Possible user-defined functions

auto&& F_lmda = MAKE_LMDA(ad::sin(x[0])*ad::cos(x[1]),
    x[2] + x[3] * x[4],
    w[0] + w[1]
);

auto&& G_lmda = MAKE_LMDA(
    ad::sum(x.begin(), x.end(), [](auto const& var)
{return ad::sin(var); }),
    w[0] * w[0] - ad::sum(x.begin(), x.end(), [](auto const& var)
{return ad::cos(var); })
);

auto&& H_lmda = MAKE_LMDA(x[0] * x[4]);

auto&& PHI_lmda = MAKE_LMDA(
    ad::sin(x[0])*ad::cos(ad::exp(x[1])) + ad::log(x[0]) - x[1],
    ad::sin(w[0]) - ad::sum(x.begin(), x.end(), [](auto const& var)
{return ad::cos(var) * ad::exp(var); }),
    ad::sin(w[1]) + ad::sum(x.begin(), x.end(), [](auto const& var)
{return ad::sin(var) * ad::exp(var); }),
    ad::sin(w[2]) + ad::prod(x.begin(), x.end(), [](auto const& var)
{return ad::cos(var); }),
    ad::sin(w[3]) + ad::sum(x.begin(), x.end(), [](auto const& var)
{return ad::sin(var) * ad::exp(var); }),
    ad::sin(w[4]) + ad::sum(x.begin(), x.end(), [](auto const& var)
{return ad::cos(var) * ad::log(var); }),
    ad::sin(w[5]) + ad::sum(x.begin(), x.end(), [](auto const& var)
{return ad::sin(var) * ad::exp(var); })
);

auto&& F = ad::make_function<double>(F_lmda);
auto&& G = ad::make_function<double>(G_lmda);
auto&& H = ad::make_function<double>(H_lmda);
auto&& PHI = ad::make_function<double>(PHI_lmda);

// GTest user-defined functions
template <class Matrix, class Iter>
void f_test(Matrix const& res, size_t i, Iter begin)
{
    EXPECT_DOUBLE_EQ(res(i, 0), std::cos(*begin)*std::cos(*std::next(begin)));
    EXPECT_DOUBLE_EQ(res(i, 1), -std::sin(*begin)*std::sin(*std::next(begin)));
    EXPECT_DOUBLE_EQ(res(i, 2), 1);
    for (size_t j = 0; j < 3; ++j) ++begin;
    EXPECT_DOUBLE_EQ(res(i, 4), *begin);
    EXPECT_DOUBLE_EQ(res(i, 3), *(++begin));
}

template <class Matrix, class Iter>
void g_test(Matrix const& res, size_t i, Iter begin)
{
    auto it = begin;
    using T = typename std::iterator_traits<Iter>::value_type;
    T sum = static_cast<T>(0);
    for (size_t j = 0; j < res.n_cols; ++it, ++j)
        sum += std::sin(*it);
    for (size_t j = 0; j < res.n_cols; ++begin, ++j)
        EXPECT_DOUBLE_EQ(res(i, j), 2 * sum*std::cos(*begin) + std::sin(*begin));
}

template <class Matrix, class Iter>
void h_test(Matrix const& res, size_t i, Iter begin)
{
    EXPECT_DOUBLE_EQ(res(i, 4), *begin);
    EXPECT_DOUBLE_EQ(res(i, 1), 0);
    EXPECT_DOUBLE_EQ(res(i, 2), 0);
    EXPECT_DOUBLE_EQ(res(i, 3), 0);
    for (size_t j = 0; j < 4; ++j) ++begin;
    EXPECT_DOUBLE_EQ(res(i, 0), *begin);
}

// ================================================================================================

    // Scalar Function test
    // With arma
template < bool arma_bool = true, class Iter, class F
    , class matrix_type = arma::Mat<double>//typename std::conditional<arma_bool, arma::Mat<double>, utils::array2d<double>>::type
>
    void test_scalar(Iter begin, Iter end, F& f)
{
    using namespace ad;
    auto&& expr = f(begin, end);
    autodiff(expr);
    matrix_type res;
    jacobian(res, f);
    f_test(res, 0, begin);
}

// Vector Function Test
// With arma
template <bool arma_bool = true, class Iter, class F
    , class matrix_type = arma::Mat<double>//typename std::conditional<arma_bool, arma::Mat<double>, utils::array2d<double>>::type
>
    void test_vector(Iter begin, Iter end, F& f, bool arma = true)
{
    using namespace ad;
    auto&& expr = f(begin, end);
    autodiff(expr);
    matrix_type res;
    jacobian(res, f);
    f_test(res, 0, begin);
    g_test(res, 1, begin);
    h_test(res, 2, begin);
}

// ================================================================================================


// Scalar Function f:R^n -> R
TEST(adeval, function_scalar) {
    using namespace ad;
    double x[] = { 0.1, 2.3, -1., 4.1, -5.21 };
    double y[] = { 2.1, 5.3, -1.23, 0.0012, -5.13 };
    auto&& F = make_function<double>(F_lmda);
    // We try both to see if Function member variable Vector is correctly
    // cleared and re-reserve capacity
    // with x
    test_scalar(x, x + 5, F);
    // with y
    test_scalar(y, y + 5, F);
}

// Scalar function with constant
TEST(adeval, function_with_constant)
{
    using namespace ad;
    double x[] = { 0.1, 2.3, -1., 4.1, -5.21 };

    auto&& F = make_function<double>(
        [](auto& x, auto& w) {
        return std::make_tuple(
            ad::sin(x[0])*ad::cos(x[1]),
            x[2] + x[3] * x[4],
            w[0] + w[1] + ad::constant(3.14)
        );
    }
    );
    // Gradient should not change from before
    test_scalar(x, x + 5, F);

    // TEST BS
    //Vec<double> X(5);
    //Vec<double> W(3);
    //auto expr = F_lmda(X, W);
    //auto&& woah = core::details::glue_many(W, expr);
    //auto&& res = core::details::glue_many(
    //	W[0] = std::get<0>(expr), W[1] = std::get<1>(expr), W[2] = std::get<2>(expr)
    //);

    // Function values should differ by 1
    auto&& F_comp = make_function<double>(F_lmda);
    auto&& expr = F(x, x + 5);
    auto&& expr_comp = F_comp(x, x + 5);
    autodiff(expr);
    autodiff(expr_comp);
    EXPECT_DOUBLE_EQ(expr.w - expr_comp.w, 3.14);
}

// Vector Function f:R^n -> R^m
TEST(adeval, function_vector) {
    using namespace ad;
    double x[] = { 0.1, 2.3, -1., 4.1, -5.21 };
    double y[] = { 2.1, 5.3, -1.23, 0.0012, -5.13 };

    auto&& F_long = make_function<double>(F_lmda, G_lmda, H_lmda);

    test_vector(x, x + 5, F_long);
    test_vector(y, y + 5, F_long);

    // Check if F is properly copied into F_long
    EXPECT_EQ(F.x.size(), 0);
    // Check if F properly contains tuple of scalar functions
    EXPECT_EQ(std::get<0>(F_long.tup).x.size(), 5);
}

// Complex Vector Function
TEST(adeval, function_vector_complex) {
    using namespace ad;
    constexpr size_t n = 1000;
    std::vector<double> x;
    std::default_random_engine gen;
    std::normal_distribution<double> dist(0.0, 1.0);

    for (size_t i = 0; i < n; ++i)
        x.push_back(dist(gen));

    auto&& F_long = make_function<double>(
        F_lmda, G_lmda, H_lmda, F_lmda, G_lmda, H_lmda, F_lmda, G_lmda, H_lmda, F_lmda,
        F_lmda, G_lmda, H_lmda, F_lmda, G_lmda, H_lmda, F_lmda, G_lmda, H_lmda, F_lmda,
        PHI_lmda, PHI_lmda, PHI_lmda, PHI_lmda, PHI_lmda
        );
    test_vector(x.begin(), x.end(), F_long);
}

// Jacobian for lambda
TEST(adeval, jacobian_lambda) {
    using namespace ad;
    double x[] = { 0.1, 2.3, -1., 4.1, -5.21 };
    arma::Mat<double> jacobi;
    jacobian<double>(jacobi, x, x + 5, F_lmda);
    f_test(jacobi, 0, x);
}

} // end namespace
