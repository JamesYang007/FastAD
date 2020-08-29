#include <fastad>
#include <benchmark/benchmark.h>

#ifdef USE_ADEPT
#include <adept_arrays.h>
#endif

// Finite-difference

static inline double f_test1(const std::vector<double>& x)
{
    double sum = 0.;
    for (size_t i = 0; i < x.size(); ++i) {
        sum += x[i];
    }

    double w0 = x[0] * x[1] - x[2] * sin(x[0]);
    double w1 = x[1] * w0 - cos(w0) + sum;
    return w1 + exp(w1 - w0);
}

static void BM_test1_fd(benchmark::State& state)
{
    constexpr double h = 1e-10;
    for (auto _ : state) {
        std::vector<double> x(100);
        for (size_t i = 0; i < x.size(); ++i) {
            x[i] = i / 100.;
        }
        double f = f_test1(x);
        for (size_t i = 0; i < x.size(); ++i) {
            x[i] += h;
            double f_h = f_test1(x);
            double dfdx_i = (f_h - f) / h;
            benchmark::DoNotOptimize(dfdx_i);
            x[i] -= h;
        }
    }
}

BENCHMARK(BM_test1_fd);

// FastAD
static void BM_test1_fastad(benchmark::State& state)
{
    using namespace ad;
    std::vector<Var<double>> x;
    for (size_t i = 0; i < 100; ++i) {
        x.emplace_back(i / 100.);
    }
    std::vector<Var<double>> w(3);
    auto expr = ad::bind(
                (w[0] = x[0] * x[1] - x[2] * sin(x[0]),
                 w[1] = x[1] * w[0] - cos(w[0]) + 
                        ad::sum(x.begin(), x.end(), [](const auto& xi) {return xi;}),
                 w[2] = w[1] + ad::exp(w[1] - w[0])) 
    );

    for (auto _ : state) {
        autodiff(expr);
        benchmark::DoNotOptimize(expr);
    }
}

BENCHMARK(BM_test1_fastad);

#ifdef USE_ADEPT

// Adept
static void BM_test1_adept(benchmark::State& state)
{
    using namespace adept;
    for (auto _ : state) {
        Stack stack;
        aVector x(100);
        for (size_t i = 0; i < 100; ++i) {
            x << i / 100.;
        }
        stack.new_recording();
        aReal w_0 = x[0] * x[1] - x[2] * sin(x[0]);
        aReal w_1 = x[1] * w_0 - cos(w_0) + sum(x);
        aReal J = w_1 + exp(w_1 - w_0);
        J.set_gradient(1.);
        stack.reverse();
        benchmark::ClobberMemory();
    }
}

BENCHMARK(BM_test1_adept);

#endif
