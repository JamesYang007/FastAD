#include "ceres/ceres.h"
#include "fastad"
#include "glog/logging.h"
#include <iostream>

using namespace ad;
// A3*s(A2*s(A1*x+b1)+b2+(A1*x+b1))+b3
struct NN {
    Eigen::MatrixXd X, y;
    NN(const Eigen::MatrixXd &X, const Eigen::VectorXd &y)
        : X(X), y(y){

                };
    /*VectorXd pred(const double *parm, const MatrixXd& X, const VectorXd&y) {

        };*/
    double loss(const double *parm, double *grad) {
        Eigen::Map<Eigen::VectorXd> g(grad, 25);
        g.setZero();

        VarView<double, mat> A1(const_cast<double *>(parm), grad, 3, 2);
        // std::cout << A1.get() << std::endl;
        VarView<double, mat> b1(const_cast<double *>(parm + 6), grad + 6, 3, 1);
        VarView<double, mat> A2(const_cast<double *>(parm + 9), grad + 9, 3, 3);
        VarView<double, mat> b2(const_cast<double *>(parm + 18), grad + 18, 3, 1);
        VarView<double, mat> A3(const_cast<double *>(parm + 21), grad + 21, 1, 3);
        VarView<double, mat> b3(const_cast<double *>(parm + 24), grad + 24, 1, 1);

        Var<double, mat> x1(3, 1), y1(3, 1), y2(3, 1), y3(1, 1), residual_norm2(1, 1); //(1, 1);
        // Var<double, mat> y3(1, 1);
        // data buffer
        Eigen::VectorXd x_row_buffer = X.row(0);
        auto xi = constant_view(x_row_buffer.data(), X.cols(), 1);
        Eigen::VectorXd y_row_buffer = y.row(0);
        auto yi = constant_view(y_row_buffer.data(), y.cols(), 1);

        // Expression
        auto expr =
            bind((x1 = dot(A1, xi) + b1, y1 = sigmoid(x1), y2 = sigmoid(dot(A2, y1) + b2) + x1,
                  y3 = dot(A3, y2) + b3, residual_norm2 = pow<2>(yi - y3)));

        // Seed
        Eigen::MatrixXd seed(1, 1);
        seed.setOnes(); // Usually seed is 1. DONT'T FORGET!

        // Loop over each row to calulate loss.
        double loss = 0;
        for (int i = 0; i < X.rows(); ++i) {
            x_row_buffer = X.row(i);
            y_row_buffer = y.row(i);

            auto f = autodiff(expr, seed.array());
            // std::cout << "pred " << i << " " << y3.get() << std::endl;
            loss += f.coeff(0);
        }
        return loss;
    };
};

// Ceres functor.
class NNfunctor : public ceres::FirstOrderFunction {
  private:
    NN &net;

  public:
    NNfunctor(NN &net) : net(net){};
    virtual bool Evaluate(const double *parameters, double *cost, double *gradient) const {
        *cost = net.loss(parameters, gradient);
        return true;
    }
    virtual int NumParameters() const { return 25; }
};
int main() {

    // Create data matrix.
    Eigen::MatrixXd X(5, 2);
    X << 1, 10, 2, 20, 3, 30, 4, 40, 5, 50;
    Eigen::VectorXd y(5);
    y << 32, 64, 96, 128, 160; // y=2*x1+3*x2

    // Generating parameter buffer and NN.
    Eigen::VectorXd parm_data(25);
    parm_data << 0.043984, 0.960126, -0.520941, -0.800526, -0.0287914, 0.635809, 0.584603,
        -0.443382, 0.224304, 0.97505, -0.824084, 0.2363, 0.666392, -0.498828, -0.781428, -0.911053,
        -0.230156, -0.136367, 0.263425, 0.841535, 0.920342, 0.65629, 0.848248, -0.748697, 0.21522;

    Eigen::VectorXd grad_data(25);
    grad_data.setZero(); // Set adjoints to zeros.
    NN net(X, y);

    // Train
    google::InitGoogleLogging("FastAD with ceres");
    ceres::GradientProblemSolver::Options options;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 1000;
    ceres::GradientProblemSolver::Summary summary;
    ceres::GradientProblem problem(new NNfunctor(net));
    ceres::Solve(options, problem, parm_data.data(), &summary);
    std::cout << summary.FullReport() << "\n";
    std::cout << "Initial x[0]: " << 0.043984 << " x[1]: " << 0.960126 << "\n";
    std::cout << "Final   x[0]: " << parm_data.coeff(0) << " x[1]: " << parm_data.coeff(1) << "\n";

    return 0;
}
