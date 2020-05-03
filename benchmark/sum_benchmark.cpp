#include <fastad_bits/vec.hpp>
#include <fastad_bits/math.hpp>
#include <fastad_bits/node.hpp>
#include <fastad_bits/eval.hpp>
#include <fastad_bits/pow.hpp>
#include <benchmark/benchmark.h>
#include <numeric>
#ifdef USE_ADEPT
#include <adept_arrays.h>
#endif

// Finite-difference method
static inline double f_test(const std::vector<double>& x)
{
    double sum = 0;
    for (const auto& xi : x) {
        sum += xi * xi;
    }
    return sum * sum + std::sin(sum);
}

static void BM_sumnode_fd(benchmark::State& state)
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
            double dfdx_i = (f_h - f) / h;
            benchmark::DoNotOptimize(dfdx_i);
            x[i] -= h;
        }
    }
}

BENCHMARK(BM_sumnode_fd);

// FastAD
static void BM_sumnode_fastad(benchmark::State& state) 
{
    using namespace ad;
    for (auto _ : state) {
        Vec<double> vec;
        for (size_t i = 0; i < 1e1; ++i) {
            vec.emplace_back(i);
        }
        Var<double> w4, w5;
        auto expr = ad::sum(vec.begin(), vec.end(), [](const auto& x) {return x * x;});
        autodiff((w4=expr, w5 = w4 * w4 + ad::sin(w4)));
        benchmark::ClobberMemory();
    }
}

BENCHMARK(BM_sumnode_fastad);

static void BM_sumnode_fastad_large_vectorized(benchmark::State& state)
{
    using namespace ad;
    constexpr size_t size = 100;
    std::vector<double> values(size);
    std::vector<double> values2(size);
    Vec<double> w(2);
    w[0].set_value(2.);
    w[1].set_value(1.);

    for (size_t i = 0; i < size; ++i) {
        values[i] = values2[i] = static_cast<double>(i);
    }

    for (auto _ : state) {
        int i = 0;
		auto expr = ad::sum(values.begin(), values.end(),
			[&, i](double v) mutable {
				if (i % values2.size() == 0) i = 0;
				auto&& expr = -ad::constant(0.5) *
					ad::pow<2>((ad::constant(v) - w[0] * ad::constant(values2[i])) / w[1]);
				++i;
				return expr;
			});
        for (int i = 0; i < 20; ++i) {
            std::for_each(w.begin(), w.end(), [](auto& x) { x.reset_adjoint(); });
            ad::autodiff(expr);
        }
		benchmark::DoNotOptimize(expr);
    }
}

BENCHMARK(BM_sumnode_fastad_large_vectorized);

#ifdef USE_ADEPT

// Adept
static void BM_sumnode_adept(benchmark::State& state)
{
    using namespace adept;    
    for (auto _ : state) {
        Stack stack;
        aVector x = {0.,1.,2.,3.,4.,5.,6.,7.,8.,9.};
        stack.new_recording();
        aReal y = sum(x * x);
        aReal J = pow(y, 2) + sin(y);
        J.set_gradient(1.);
        stack.reverse();
        benchmark::ClobberMemory();
    }
}

BENCHMARK(BM_sumnode_adept);

#endif
