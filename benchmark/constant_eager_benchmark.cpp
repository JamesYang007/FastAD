#include <fastad_bits/node.hpp>
#include <fastad_bits/math.hpp>
#include <fastad_bits/eval.hpp>
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
