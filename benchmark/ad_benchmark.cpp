#include <fastad>
#include <benchmark/benchmark.h>
#include <adept_arrays.h>

static void BM_test1(benchmark::State& state)
{
    using namespace ad;
    for (auto _ : state) {
        Vec<double> x({0,1,2});
        Vec<double> w(3);
        auto expr = (w[0] = x[0] * x[1] - x[2] * sin(x[0]),
                     w[1] = x[1] * w[0] - cos(w[0]),
                     w[2] = w[1] + exp(w[1] - w[0]));
        autodiff(expr);
        benchmark::ClobberMemory();
    }
}

BENCHMARK(BM_test1);

static void BM_adept_test1(benchmark::State& state)
{
    using namespace adept;
    for (auto _ : state) {
        Stack stack;
        aVector x = {0.,1.,2.};
        stack.new_recording();
        aReal w_0 = x[0] * x[1] - x[2] * sin(x[0]);
        aReal w_1 = x[1] * w_0 - cos(w_0);
        aReal J = w_1 + exp(w_1 - w_0);
        J.set_gradient(1.);
        stack.reverse();
        benchmark::ClobberMemory();
    }
}

BENCHMARK(BM_adept_test1);
