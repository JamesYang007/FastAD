#include <fastad>
#include <random>
#include <time.h>
#include "gtest/gtest.h"

namespace ad {
namespace core {

auto F_lmda = [](const Vec<double>& x, const Vec<double>& w) {
    return (w[0] = ad::sin(x[0])*ad::cos(x[1]),
            w[1] = x[2] + x[3] * x[4],
            w[2] = w[0] + w[1]);
};

auto G_lmda = [](const Vec<double>& x, const Vec<double>& w) {
    return (w[0] = ad::sum(x.begin(), x.end(), [](const auto& var)
                            {return ad::sin(var); }),
            w[1] = w[0] * w[0] - ad::sum(x.begin(), x.end(), [](const auto& var)
                                    {return ad::cos(var); })
            );
};

auto H_lmda = [](const Vec<double>& x, const Vec<double>& w) {
    return (w[0] = x[0] * x[4]);
};

auto PHI_lmda = [](const Vec<double>& x, const Vec<double>& w) {
    return (w[0] = ad::sin(x[0])*ad::cos(ad::exp(x[1])) + ad::exp(x[0]) - x[1],
            w[1] = ad::sin(w[0]) - ad::sum(x.begin(), x.end(), [](const auto& var)
                                    {return ad::cos(var) * ad::exp(var); }),
            w[2] = ad::sin(w[1]) + ad::sum(x.begin(), x.end(), [](const auto& var)
                                    {return ad::sin(var) * ad::exp(var); }),
            w[3] = ad::sin(w[2]) + ad::prod(x.begin(), x.end(), [](const auto& var)
                                    {return ad::cos(var); }),
            w[4] = ad::sin(w[3]) + ad::sum(x.begin(), x.end(), [](const auto& var)
                                    {return ad::sin(var) * ad::exp(var); }),
            w[5] = ad::sin(w[4]) + ad::sum(x.begin(), x.end(), [](const auto& var)
                                    {return ad::cos(var) * ad::exp(var); }),
            w[6] = ad::sin(w[5]) + ad::sum(x.begin(), x.end(), [](const auto& var)
                                    {return ad::sin(var) * ad::exp(var); })
    );
};

auto F = ad::make_exgen<double>(F_lmda);
auto G = ad::make_exgen<double>(G_lmda);
auto H = ad::make_exgen<double>(H_lmda);
auto PHI = ad::make_exgen<double>(PHI_lmda);

struct ad_fixture : ::testing::Test
{
protected:

    template <class Matrix, class Iter>
    void f_test(const Matrix& res, size_t i, Iter begin)
    {
        EXPECT_DOUBLE_EQ(res(i, 0), std::cos(*begin)*std::cos(*std::next(begin)));
        EXPECT_DOUBLE_EQ(res(i, 1), -std::sin(*begin)*std::sin(*std::next(begin)));
        EXPECT_DOUBLE_EQ(res(i, 2), 1);
        for (size_t j = 0; j < 3; ++j) ++begin;
        EXPECT_DOUBLE_EQ(res(i, 4), *begin);
        EXPECT_DOUBLE_EQ(res(i, 3), *(++begin));
    }

    template <class Matrix, class Iter>
    void g_test(const Matrix& res, size_t i, Iter begin)
    {
        auto it = begin;
        using T = typename std::iterator_traits<Iter>::value_type;
        T sum = static_cast<T>(0);
        for (size_t j = 0; j < res.n_cols(); ++it, ++j)
            sum += std::sin(*it);
        for (size_t j = 0; j < res.n_cols(); ++begin, ++j)
            EXPECT_DOUBLE_EQ(res(i, j), 2 * sum*std::cos(*begin) + std::sin(*begin));
    }

    template <class Matrix, class Iter>
    void h_test(const Matrix& res, size_t i, Iter begin)
    {
        EXPECT_DOUBLE_EQ(res(i, 4), *begin);
        EXPECT_DOUBLE_EQ(res(i, 1), 0);
        EXPECT_DOUBLE_EQ(res(i, 2), 0);
        EXPECT_DOUBLE_EQ(res(i, 3), 0);
        for (size_t j = 0; j < 4; ++j) ++begin;
        EXPECT_DOUBLE_EQ(res(i, 0), *begin);
    }

    // Scalar Function test
    template <class Iter, class F>
    void test_scalar(Iter begin, Iter end, F& f)
    {
        using T = typename std::iterator_traits<Iter>::value_type;
        ad::Mat<T> res;
        jacobian(res, begin, end, f);
        f_test(res, 0, begin);
    }

    // Vector Function Test
    template <class Iter, class... Fs>
    void test_vector(Iter begin, Iter end, Fs&... fs)
    {
        using T = typename std::iterator_traits<Iter>::value_type;
        ad::Mat<T> res;
        jacobian(res, begin, end, fs...);
        f_test(res, 0, begin);
        g_test(res, 1, begin);
        h_test(res, 2, begin);
    }


};

// Scalar Function f:R^n -> R
TEST_F(ad_fixture, function_scalar) {
    double x[] = { 0.1, 2.3, -1., 4.1, -5.21 };
    double y[] = { 2.1, 5.3, -1.23, 0.0012, -5.13 };
    test_scalar(x, x + 5, F_lmda);
    test_scalar(y, y + 5, F_lmda);
}

// Scalar function with constant
TEST_F(ad_fixture, function_with_constant)
{
    double x[] = { 0.1, 2.3, -1., 4.1, -5.21 };

    auto f_lmda = [](const Vec<double>& x, const Vec<double>& w) {
        return (w[0] = ad::sin(x[0])*ad::cos(x[1]),
                w[1] = x[2] + x[3] * x[4],
                w[2] = w[0] + w[1] + ad::constant(3.14));
    };

    // Gradient should not change from before
    test_scalar(x, x + 5, f_lmda);

    // Function values should differ by 1
    auto f_gen = make_exgen<double>(f_lmda);
    Vec<double> v(x, x + 5);
    double res = evaluate(std::get<0>(f_gen.generate(v)));
    res -= evaluate(std::get<0>(F.generate(v)));    // subtract away original generator value
    EXPECT_DOUBLE_EQ(res, 3.14);
}

// Vector Function f:R^n -> R^m
TEST_F(ad_fixture, function_vector) {
    double x[] = { 0.1, 2.3, -1., 4.1, -5.21 };
    double y[] = { 2.1, 5.3, -1.23, 0.0012, -5.13 };
    test_vector(x, x + 5, F_lmda, G_lmda, H_lmda);
    test_vector(y, y + 5, F_lmda, G_lmda, H_lmda);
}

// Vector Function f:R^n -> R^m
// same lambda
TEST_F(ad_fixture, function_vector_same_lmda) {
    double x[] = { 0.1, 2.3, -1., 4.1, -5.21 };
    test_vector(x, x + 5, F_lmda, G_lmda, H_lmda, F_lmda);
}

// Complex Vector Function
TEST_F(ad_fixture, function_vector_complex) {
    constexpr size_t n = 10;
    std::vector<double> x;
    std::default_random_engine gen;
    std::normal_distribution<double> dist(0., 1.);

    for (size_t i = 0; i < n; ++i) {
        x.push_back(dist(gen));
    }

    test_vector(x.begin(), x.end(), 
        F_lmda, G_lmda, H_lmda, F_lmda, G_lmda, PHI_lmda, PHI_lmda
            );
}

} // namespace core
} // namespace ad
