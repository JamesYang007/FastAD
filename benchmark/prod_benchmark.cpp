#include <fastad_bits/reverse/core/var.hpp>
#include <fastad_bits/reverse/core/eq.hpp>
#include <fastad_bits/reverse/core/unary.hpp>
#include <fastad_bits/reverse/core/binary.hpp>
#include <fastad_bits/reverse/core/eval.hpp>
#include <fastad_bits/reverse/core/pow.hpp>
#include <fastad_bits/reverse/core/prod.hpp>
#include <benchmark/benchmark.h>
#ifdef USE_ADEPT
#include <adept_arrays.h>
#endif

// Finite-difference

static inline double f_test(const std::vector<double>& x)
{
    double prod = 1.;
    for (const auto& xi : x) {
        prod *= xi * xi;
    }
    return prod * prod + std::cos(prod);
}

static void BM_prod_fd(benchmark::State& state)
{
    constexpr double h = 1e-10;
    std::vector<double> x(10);
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] = i;
    }
    for (auto _ : state) {
        double f = f_test(x);
        for (size_t i = 0; i < x.size(); ++i) {
            x[i] += h;
            double f_h = f_test(x);
            double dfdx_i = (f_h - f)/h;
            benchmark::DoNotOptimize(dfdx_i);
            x[i] -= h;
        }
    }
}

BENCHMARK(BM_prod_fd);

// FastAD
static void BM_prod_fastad(benchmark::State& state) 
{
    using namespace ad;
    std::vector<Var<double>> vec; 
    for (size_t i = 0; i < 1e1; ++i) {
        vec.emplace_back(i);
    }
    Var<double> w4, w5;
    auto prod_expr = ad::prod(vec.begin(), vec.end()
        , [](const auto& x) {
            return x * x;
        });
    auto expr = ad::bind((w4 = prod_expr, w5 = w4 * w4 + ad::cos(w4)));

    for (auto _ : state) {
        autodiff(expr);
        benchmark::ClobberMemory();
    }
}

BENCHMARK(BM_prod_fastad);

#ifdef USE_ADEPT

// Adept
static void BM_prod_adept(benchmark::State& state)
{
    using namespace adept;    
    for (auto _ : state) {
        Stack stack;
        aVector x = {0.,1.,2.,3.,4.,5.,6.,7.,8.,9.};
        stack.new_recording();
        aReal y = product(x * x);
        aReal J = pow(y, 2) + cos(y);
        J.set_gradient(1.);
        stack.reverse();
        benchmark::ClobberMemory();
    }
}

BENCHMARK(BM_prod_adept);

#endif

