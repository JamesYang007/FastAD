#include "fastad"
#include <iostream>

int main() {
    using namespace ad;
    Eigen::MatrixXd x_data(2, 2);
    x_data << -0.1, 0.1, 0.2, 1;
    Eigen::MatrixXd x_adj(2, 2);
    x_adj.setZero();

    VarView<double, mat> x(x_data.data(), x_adj.data(), 2, 2);
    auto expr = bind(sigmoid(x));
    Eigen::MatrixXd seed(2, 2);
    seed.setOnes();

    auto sigmoid_f = autodiff(expr, seed.array());
    std::cout << sigmoid_f << std::endl;
    std::cout << x_adj << std::endl;

    std::cout << "--------" << std::endl;
    seed.setOnes();
    x_adj.setZero();
    auto tanh_expr = bind(tanh(x));
    auto tanh_f = autodiff(tanh_expr, seed.array());
    // std::cout << x_data << std::endl;
    std::cout << tanh_f << std::endl;
    std::cout << x_adj << std::endl;
    std::cout << Eigen::tanh(x_data.array()) << std::endl;
    std::cout << 1 - Eigen::tanh(x_data.array()).pow(2) << std::endl;
}