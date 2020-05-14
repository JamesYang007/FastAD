#include <fastad_bits/node.hpp>
#include <fastad_bits/math.hpp>
#include <fastad_bits/eval.hpp>
#include <fastad_bits/pow.hpp>
#include <benchmark/benchmark.h>
#include <vector>
#include <numeric>

static void BM_repeated_constants(benchmark::State& state)
{
    auto c = ad::log(ad::constant(1.0));

    // create vector of copies of expressions
    using value_t = std::decay_t<decltype(c)>;
    std::vector<value_t> v(100000, c);

    for (auto _ : state) {
        for (auto& x : v) {
            ad::evaluate(x);
        }
        benchmark::DoNotOptimize(v);
    }
}

BENCHMARK(BM_repeated_constants);

static void BM_normal_repeated_stddev(benchmark::State& state)
{
    using namespace ad;
    constexpr size_t size = 1000;
    std::vector<double> values(size);
    std::vector<double> values2(size);
    Var<double> w(2.);

    for (size_t i = 0; i < size; ++i) {
        values[i] = values2[i] = static_cast<double>(i);
    }

    int i = 0;
    auto expr = ad::sum(values.begin(), values.end(),
        [&, i](double v) mutable {
            auto&& expr = -ad::constant(0.5) *
                ad::pow<2>((ad::constant(v) - w * ad::constant(values2[i])) / ad::constant(2.)) -
                ad::log(ad::constant(2.));
            ++i;
            return expr;
        });

    for (auto _ : state) {
        ad::autodiff(expr);
		benchmark::DoNotOptimize(expr);
    }
}

BENCHMARK(BM_normal_repeated_stddev);
