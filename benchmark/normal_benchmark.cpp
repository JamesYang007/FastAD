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

static void BM_normal_adj_log_pdf(benchmark::State& state)
{
    using namespace ad;
    using value_t = double;

    size_t size = state.range(0);

    VarView<value_t, vec> x(nullptr, nullptr, size);
    VarView<value_t, vec> mu(nullptr, nullptr, size);
    VarView<value_t, selfadjmat> sigma(nullptr, nullptr, size, size);

    size_t tot_size = x.size() + mu.size() + sigma.size();
    std::vector<double> val(tot_size);
    std::vector<double> adj(tot_size);

    auto val_next = x.bind(val.data());
    auto adj_next = x.bind_adj(adj.data());
    val_next = mu.bind(val_next);
    adj_next = mu.bind_adj(adj_next);
    val_next = sigma.bind(val.data());
    adj_next = sigma.bind_adj(adj_next);

    x.get().Random();
    mu.get().Random();

    bool is_pos_def_ = false;

    Eigen::MatrixXd lower(sigma.rows(), sigma.cols());

    while (!is_pos_def_) {
        lower = Eigen::MatrixXd::Random(sigma.rows(), sigma.cols());
        lower.template triangularView<Eigen::Upper>().setZero();
        lower.diagonal().array() = 10;
        sigma.get() = lower * lower.transpose();     // make positive definite

        Eigen::LLT<Eigen::Map<Eigen::MatrixXd>> llt(sigma.get());
        is_pos_def_ = (llt.info() == Eigen::Success);
    }
   
    if (!is_pos_def_) {
        std::cerr << "Warning: some covariance matrix was not positive definite" 
                  << std::endl;
    }

    auto expr = normal_adj_log_pdf(x, mu, sigma);
    std::vector<double> tmp(expr.bind_size());
    expr.bind(tmp.data());

    for (auto _ : state) {
        ad::autodiff(expr);
		benchmark::DoNotOptimize(expr);
    }
}

BENCHMARK(BM_normal_adj_log_pdf)->Arg(10)
                                ->Arg(50)
                                ->Arg(100)
                                ->Arg(500)
                                ->Arg(1000)
                                ->Arg(2000)
                                ->Arg(3000)
                                ->Arg(4000);
