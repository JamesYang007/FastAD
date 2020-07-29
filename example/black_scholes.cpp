#include <fastad>
#include <iostream>

enum class option_type {
    call, put
};

// Standard Normal CDF
template <class T>
inline auto Phi(const T& x)
{
    return 0.5 * (ad::erf(x / std::sqrt(2.)) + 1.);
}

// Generates expression that computes Black-Scholes option price
template <option_type cp, class Price, class Cache>
auto black_scholes_option_price(const Price& S,
                                double K,
                                double sigma,
                                double tau,
                                double r,
                                Cache& cache)
{
    cache.resize(3);
    double PV = K * std::exp(-r * tau);
    auto common_expr = (
            cache[0] = ad::log(S / K),
            cache[1] = (cache[0] + ((r + sigma * sigma / 2.) * tau)) / 
                            (sigma * std::sqrt(tau)),
            cache[2] = cache[1] - (sigma * std::sqrt(tau))
    );
    if constexpr (cp == option_type::call) {
        return (common_expr,
                Phi(cache[1]) * S - Phi(cache[2]) * PV);
    } else {
        return (common_expr,
                Phi(-cache[2]) * PV - Phi(-cache[1]) * S);
    }
}

int main()
{
    double K = 100.0;        
    double sigma = 5;        
    double tau = 30.0 / 365; 
    double r = 1.25 / 100;   
    ad::Var<double> S(105);  
    std::vector<ad::Var<double>> cache;

    auto call_expr = ad::bind(
            black_scholes_option_price<option_type::call>(
                S, K, sigma, tau, r, cache));

    double call_price = ad::autodiff(call_expr);

    std::cout << call_price << std::endl;
    std::cout << S.get_adj() << std::endl;

    // reset adjoints before differentiating again
    S.reset_adj();
    for (auto& c : cache) c.reset_adj();

    auto put_expr = ad::bind(
            black_scholes_option_price<option_type::put>(
                S, K, sigma, tau, r, cache));

    double put_price = ad::autodiff(put_expr);

    std::cout << put_price << std::endl;
    std::cout << S.get_adj() << std::endl;

    return 0;
}
