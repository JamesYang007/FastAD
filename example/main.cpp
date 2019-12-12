#include <fastad>
#include <iostream>

#define FLOAT_PRINT_PRECISION 5

// Forward AD
void forward()
{
	using namespace ad;
	double x1 = -0.201, x2 = 1.2241;
	ForwardVar<double> w1(x1), w2(x2);

	// Take partial w.r.t. w1
	w1.get_adjoint() = 1;
	ForwardVar<double> w3 = w1 * sin(w2);
	auto w4 = w3 + w1 * w2;
	auto w5 = exp(w4*w3);

	// Partial w.r.t. w1
	std::cout << "f(x, y) = exp((x * sin(y) + x * y) * x * sin(y))\n"
            << "df/dx = " << w5.get_adjoint() << std::endl;
}

// Reverse AD
void reverse_simple()
{
	using namespace ad;
	double x1 = -0.201, x2 = 1.2241;
	Var<double> w1(x1);
	Var<double> w2(x2);

	Var<double> w3;
	Var<double> w4;
	Var<double> w5;

	auto expr = (
		w3 = w1 * sin(w2)
		, w4 = w3 + w1 * w2
		, w5 = exp(w4*w3)
		);

	autodiff(expr);

	std::cout << "f(x, y) = exp((x * sin(y) + x * y) * x * sin(y))\n"
            << "df/dx = " << w1.get_adjoint() << std::endl
            << "df/dy = " << w2.get_adjoint() << std::endl;
}

void reverse_vec()
{
	using namespace ad;
	Vec<double> x(0);
	Vec<double> w(3);
	double x_val[] = { -0.201, 1.2241 };
    for (size_t i = 0; i < 2; ++i) {
		x.emplace_back(x_val[i]);
    }

	auto expr = (
		w[0] = x[0] * sin(x[1])
		, w[1] = w[0] + x[0] * x[1]
		, w[2] = exp(w[1] * w[0])
		);

	autodiff(expr);

	std::cout << "f(x, y) = exp((x * sin(y) + x * y) * x * sin(y))\n"
            << "df/dx = " << x[0].get_adjoint() << std::endl
            << "df/dy = " << x[1].get_adjoint() << std::endl;
}

void reverse_function()
{
	using namespace ad;
	Vec<double> x = { -0.201, 1.2241 };
	auto F_lmda = [](const auto& x, const auto& w) {
		return (w[0] = x[0] * ad::sin(x[1]),
                w[1] = w[0] + x[0] * x[1],
                w[2] = ad::exp(w[1] * w[0]));
	};

	auto gen = make_exgen<double>(F_lmda);
	autodiff(gen.generate(x));

	std::cout << "f(x, y) = exp((x * sin(y) + x * y) * x * sin(y))\n"
            << "df/dx = " << x[0].get_adjoint() << std::endl
            << "df/dy = " << x[1].get_adjoint() << std::endl;
}

void reverse_jacobian()
{
	using namespace ad;
	auto F_lmda = [](const auto& x, const auto& w) {
		return (w[0] = x[0] * ad::sin(x[1]),
                w[1] = w[0] + x[0] * x[1],
                w[2] = ad::exp(w[1] * w[0]));
	};
	double x_val[] = { -0.201, 1.2241 };		// substitute for any data structure that is iterable
	Mat<double> jacobi;					// substitute for a 2D-array data structure
												// (more information in documentation)
	jacobian(jacobi, x_val, x_val + 2, F_lmda);
	std::cout << "f(x, y) = exp((x * sin(y) + x * y) * x * sin(y))" << std::endl;
	jacobi.print_at_precision("Jacobian of f(x, y)", FLOAT_PRINT_PRECISION);
}

void reverse_vector()
{
	using namespace ad;
	auto F_lmda = [](const auto& x, const auto& w) {
		return (w[0] = x[0] * ad::sin(x[1]),
                w[1] = w[0] + x[0] * x[1],
                w[2] = ad::exp(w[1] * w[0]));
	};
	auto G_lmda = [](const auto& x, const auto& w) {
		return (w[0] = x[0] + ad::exp(ad::sin(x[1])),
		        w[1] = w[0] * w[0] * x[1]);
    };

	double x_val[] = { -0.201, 1.2241 };
	Mat<double> jacobi;

	jacobian(jacobi, x_val, x_val + 2, F_lmda, G_lmda); 
	std::cout << "f(x, y) = exp((x * sin(y) + x * y) * x * sin(y))\n"
            << "g(x, y) = (x + exp(sin(y)))^2 * y" << std::endl;
	jacobi.print_at_precision("Jacobian of (f(x, y), g(x, y))", FLOAT_PRINT_PRECISION);
}

// Hessian
void hessian()
{
	using namespace ad;
	auto F_lmda = [](const auto& x, const auto& w) {
		return (w[0] = x[0] * ad::sin(x[1]),
                w[1] = w[0] + x[0] * x[1],
                w[2] = ad::exp(w[1] * w[0]));
	};
	double x_val[] = { -0.201, 1.2241 };
	Mat<double> hess;
	Mat<double> jacobi;

	// Computes Hessian and stores into "hess"
	hessian(hess, x_val, x_val + 2, F_lmda);
	// Computes Hessian and stores Hessian into "hess" and Jacobian into "jacobi"
	hessian(hess, jacobi, x_val, x_val + 2, F_lmda);

	std::cout << "f(x, y) = exp((x * sin(y) + x * y) * x * sin(y))" << std::endl;
	hess.print_at_precision("Hessian of f(x, y)", FLOAT_PRINT_PRECISION);
	jacobi.print_at_precision("Jacobian of f(x, y)", FLOAT_PRINT_PRECISION);
}

int main()
{
	forward();
	reverse_simple();
	reverse_vec();
	reverse_function();

	reverse_jacobian();
	reverse_vector();
	hessian();

	return 0;
}
