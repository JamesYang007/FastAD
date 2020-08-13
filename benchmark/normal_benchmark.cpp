#include <fastad_bits/reverse/core/math.hpp>
#include <fastad_bits/reverse/core/var.hpp>
#include <fastad_bits/reverse/core/eval.hpp>
#include <fastad_bits/reverse/core/eq.hpp>
#include <fastad_bits/reverse/core/pow.hpp>
#include <fastad_bits/reverse/core/sum.hpp>
#include <fastad_bits/reverse/stat/normal.hpp>
#include <benchmark/benchmark.h>
#include <numeric>
#include <iostream>

struct normal_fixture : benchmark::Fixture 
{
    using value_t = double;

    template <class SigmaType>
    void make_cov(SigmaType& sigma) 
    {
        bool is_pos_def_ = false;

        Eigen::MatrixXd lower(sigma.rows(), sigma.cols());

        while (!is_pos_def_) {
            lower = Eigen::MatrixXd::Random(sigma.rows(), sigma.cols());
            lower.template triangularView<Eigen::Upper>().setZero();
            lower.diagonal().array() = 10;
            sigma.get() = lower * lower.transpose();     // make positive definite

            Eigen::LLT<Eigen::MatrixXd> llt(sigma.get());
            is_pos_def_ = (llt.info() == Eigen::Success);
        }
       
        if (!is_pos_def_) {
            std::cerr << "Warning: some covariance matrix was not positive definite" 
                      << std::endl;
        }
    }
};

BENCHMARK_DEFINE_F(normal_fixture, BM_normal_adj_log_pdf)(benchmark::State& state)
{
    using namespace ad;

    size_t size = state.range(0);

    VarView<value_t, vec> x(nullptr, nullptr, size);
    VarView<value_t, vec> mu(nullptr, nullptr, size);
    VarView<value_t, selfadjmat> sigma(nullptr, nullptr, size, size);

    size_t tot_size = x.size() + mu.size() + sigma.size();
    std::vector<double> val(tot_size, 0);
    std::vector<double> adj(tot_size, 0);

    auto val_next = x.bind(val.data());
    auto adj_next = x.bind_adj(adj.data());
    val_next = mu.bind(val_next);
    adj_next = mu.bind_adj(adj_next);
    val_next = sigma.bind(val_next);
    adj_next = sigma.bind_adj(adj_next);

    x.get().Random();
    mu.get().Random();
    make_cov(sigma);

    auto expr = ad::bind(normal_adj_log_pdf(x, mu, sigma));

    for (auto _ : state) {
        ad::autodiff(expr);
		benchmark::DoNotOptimize(expr);
    }
}

BENCHMARK_DEFINE_F(normal_fixture, BM_normal_adj_log_pdf_flat)(benchmark::State& state)
{
    using namespace ad;

    size_t size = state.range(0);

    VarView<value_t, vec> x(nullptr, nullptr, size);
    VarView<value_t, vec> mu(nullptr, nullptr, size);

    size_t tot_size = x.size() + mu.size() + (size * (size + 1))/2;
    std::vector<double> val(tot_size, 0);
    std::vector<double> adj(tot_size, 0);

    auto val_next = x.bind(val.data());
    auto adj_next = x.bind_adj(adj.data());
    val_next = mu.bind(val_next);
    adj_next = mu.bind_adj(adj_next);

    Eigen::MatrixXd mat_vals(size, size);
    VarView<value_t, selfadjmat> sigma(mat_vals.data(), 
                                       val_next, 
                                       adj_next, 
                                       size);

    x.get().Random();
    mu.get().Random();
    make_cov(sigma);

    size_t k = 0;
    for (size_t j = 0; j < sigma.cols(); ++j) {
        for (size_t i = j; i < sigma.rows(); ++i, ++k) {
            val_next[k] = sigma.get()(i,j);
        }
    }

    auto expr = ad::bind(normal_adj_log_pdf(x, mu, sigma));

    for (auto _ : state) {
        ad::autodiff(expr);
		benchmark::DoNotOptimize(expr);
    }
}

BENCHMARK_REGISTER_F(normal_fixture, 
                     BM_normal_adj_log_pdf)
    ->Arg(10)
    ->Arg(50)
    ->Arg(100)
    ->Arg(500)
    ->Arg(1000)
    ->Arg(2000)
    ->Arg(3000)
    ->Arg(4000);

BENCHMARK_REGISTER_F(normal_fixture, 
                     BM_normal_adj_log_pdf_flat)
    ->Arg(10)
    ->Arg(50)
    ->Arg(100)
    ->Arg(500)
    ->Arg(1000)
    ->Arg(2000)
    ->Arg(3000)
    ->Arg(4000);
