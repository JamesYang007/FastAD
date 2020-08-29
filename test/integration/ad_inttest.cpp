#include "gtest/gtest.h"
#include <fastad>
#include <random>
#include <array>
#include <time.h>

namespace ad {
namespace core {

auto F_lmda = [](const auto& x, const auto& w) {
    return (w[0] = ad::sin(x[0])*ad::cos(x[1]),
            w[1] = x[2] + x[3] * x[4],
            w[2] = w[0] + w[1]);
};

auto G_lmda = [](const auto& x, const auto& w) {
    return (w[0] = ad::sum(x.begin(), x.end(), [](const auto& var)
                            {return ad::sin(var); }),
            w[1] = w[0] * w[0] - ad::sum(x.begin(), x.end(), [](const auto& var)
                                    {return ad::cos(var); })
            );
};

auto H_lmda = [](const auto& x, const auto& w) {
    return (w[0] = x[0] * x[4]);
};

auto PHI_lmda = [](const auto& x, const auto& w) {
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

struct ad_fixture : ::testing::Test
{
protected:
    enum class Func : char
    { f, g, h };

    std::array<double, 5> val = { 0.1, 2.3, -1., 4.1, -5.21 };
    std::array<double, 5> adj = {0};
    std::array<ad::VarView<double>, 5> v;
    std::array<ad::Var<double>, 10> w;

    ad_fixture()
    {
        for (size_t i = 0; i < v.size(); ++i) {
            v[i].bind({val.data() + i, adj.data() + i});
        }
    }

    template <class AdjType, class ValType>
    void f_test(const ValType& val, const AdjType& adj)
    {
        EXPECT_DOUBLE_EQ(adj[0], std::cos(val[0])*std::cos(val[1]));
        EXPECT_DOUBLE_EQ(adj[1], -std::sin(val[0])*std::sin(val[1]));
        EXPECT_DOUBLE_EQ(adj[2], 1);
        EXPECT_DOUBLE_EQ(adj[3], val[4]);
        EXPECT_DOUBLE_EQ(adj[4], val[3]);
    }

    template <class AdjType, class ValType>
    void g_test(const ValType& val, const AdjType& adj)
    {
        double sum = 0;
        for (size_t j = 0; j < adj.size(); ++j)
            sum += std::sin(val[j]);
        for (size_t j = 0; j < adj.size(); ++j)
            EXPECT_DOUBLE_EQ(adj[j], 2 * sum*std::cos(val[j]) + std::sin(val[j]));
    }

    template <class AdjType, class ValType>
    void h_test(const ValType& val, const AdjType& adj)
    {
        EXPECT_DOUBLE_EQ(adj[0], val[4]);
        EXPECT_DOUBLE_EQ(adj[1], 0);
        EXPECT_DOUBLE_EQ(adj[2], 0);
        EXPECT_DOUBLE_EQ(adj[3], 0);
        EXPECT_DOUBLE_EQ(adj[4], val[0]);
    }

    // Scalar Function test
    template <class ExprType, class ValType, class AdjType>
    void test_scalar(ExprType& expr, 
                     const ValType& val, 
                     AdjType& adj,
                     Func test_func)
    {
        auto size_pack = expr.bind_cache_size();
        std::vector<double> val_buf(size_pack(0));
        std::vector<double> adj_buf(size_pack(1));
        expr.bind_cache({val_buf.data(), adj_buf.data()});
        autodiff(expr);
        switch(test_func) {
            case Func::f:
                f_test(val, adj);
                break;
            case Func::g:
                g_test(val, adj);
                break;
            case Func::h:
                h_test(val, adj);
                break;
        }
    }
};

TEST_F(ad_fixture, F_test) {
    auto expr = F_lmda(v, w);
    test_scalar(expr, val, adj, Func::f);
}

TEST_F(ad_fixture, F_with_constant_test)
{
    auto expr = (w[0] = ad::sin(v[0])*ad::cos(v[1]),
                w[1] = v[2] + v[3] * v[4],
                w[2] = w[0] + w[1] + ad::constant(3.14));
    test_scalar(expr, val, adj, Func::f);
}

TEST_F(ad_fixture, G_test) {
    auto expr = G_lmda(v, w);
    test_scalar(expr, val, adj, Func::g);
}

TEST_F(ad_fixture, H_test) {
    auto expr = H_lmda(v, w);
    test_scalar(expr, val, adj, Func::h);
}

// Complex Vector Function
TEST_F(ad_fixture, function_vector_complex) {
    constexpr size_t n = 1000;
    std::vector<Var<double>> x;
    std::default_random_engine gen;
    std::normal_distribution<double> dist(0., 1.);

    x.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        x.emplace_back(dist(gen));
    }

    auto expr = ad::bind(PHI_lmda(x, w));
    autodiff(expr);
}

} // namespace core
} // namespace ad
