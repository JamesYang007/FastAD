#include <fastad_bits/node.hpp>
#include <fastad_bits/prod.hpp>
#include <fastad_bits/math.hpp>
#include <fastad_bits/vec.hpp>
#include <fastad_bits/eval.hpp>
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
    for (auto _ : state) {
        std::vector<double> x(10);
        for (size_t i = 0; i < x.size(); ++i) {
            x[i] = i;
        } 
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
    for (auto _ : state) {
        Vec<double> vec; 
        for (size_t i = 0; i < 1e1; ++i) {
            vec.emplace_back(i);
        }
        Var<double> w4, w5;
        auto expr = ad::prod(vec.begin(), vec.end()
            , [](const Vec<double>::value_type& x) {
                return x * x;
            });
        autodiff((w4 = expr, w5 = w4 * w4 + ad::cos(w4)));
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

