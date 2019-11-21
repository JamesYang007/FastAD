#include <fastad/autodiff.hpp>
#include <armadillo>

// Forward AD

void forward()
{
	using namespace ad;
	double x1 = -0.201, x2 = 1.2241;
	ForwardVar<double> w1(x1), w2(x2);

	// Take partial w.r.t. w1
	w1.df = 1;
	ForwardVar<double> w3 = w1 * sin(w2);
	auto w4 = w3 + w1 * w2;
	auto w5 = exp(w4*w3);

	// Partial w.r.t. w1
	std::cout << w5.df << std::endl;
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

	std::cout << w1.df << std::endl; // partial derivative w.r.t. w1
	std::cout << w2.df << std::endl; // partial derivative w.r.t. w2
}

void reverse_vec()
{
	using namespace ad;
	Vec<double> x(0);
	Vec<double> w(3);
	double x_val[] = { -0.201, 1.2241 };
	for (size_t i = 0; i < 2; ++i)
		x.emplace_back(x_val[i]);

	auto expr = (
		w[0] = x[0] * sin(x[1])
		, w[1] = w[0] + x[0] * x[1]
		, w[2] = exp(w[1] * w[0])
		);

	autodiff(expr);

	std::cout << x[0].df << std::endl; // partial derivative w.r.t. w1
	std::cout << x[1].df << std::endl; // partial derivative w.r.t. w2
}

void reverse_function()
{
	using namespace ad;
	double x_val[] = { -0.201, 1.2241 };
	auto&& F_lmda = [](auto& x, auto& w) {
		return std::make_tuple(
			x[0] * ad::sin(x[1]),
			w[0] + x[0] * x[1],
			ad::exp(w[1] * w[0])
		);
	};
	auto F = make_function<double>(F_lmda);
	autodiff(F(x_val, x_val + 2));

	std::cout << F.x[0].df << std::endl;
	std::cout << F.x[1].df << std::endl;
}

void reverse_jacobian()
{
	using namespace ad;
	auto F_lmda = MAKE_LMDA(
		x[0] * ad::sin(x[1]),
		w[0] + x[0] * x[1],
		ad::exp(w[1] * w[0])
	);
	double x_val[] = { -0.201, 1.2241 };		// substitute for any data structure that is iterable
	auto F = make_function<double>(F_lmda);
	autodiff(F(x_val, x_val + 2));
	arma::Mat<double> jacobi;					// substitute for a 2D-array data structure
												// satisfying certain properties
												// (more information in documentation)
	jacobian(jacobi, F);
	jacobi.print("Jacobian");					// armadillo feature
}

void reverse_jacobian_2()
{
	using namespace ad;
	auto F_lmda = MAKE_LMDA(
		x[0] * ad::sin(x[1]),
		w[0] + x[0] * x[1],
		ad::exp(w[1] * w[0])
	);
	double x_val[] = { -0.201, 1.2241 };		// substitute for any data structure that is iterable
	arma::Mat<double> jacobi;					// substitute for a 2D-array data structure
												// satisfying certain properties
												// (more information in documentation)
	jacobian<double>(jacobi, x_val, x_val + 2, F_lmda);
	jacobi.print("Jacobian");					// armadillo feature
}

// Vector Function
// Function Object
void reverse_vector()
{
	using namespace ad;
	auto F_lmda = MAKE_LMDA(
		x[0] * ad::sin(x[1]),
		w[0] + x[0] * x[1],
		ad::exp(w[1] * w[0])
	);
	auto G_lmda = MAKE_LMDA(
		x[0] + ad::exp(ad::sin(x[1])),
		w[0] * w[0] * x[1]
	);
	double x_val[] = { -0.201, 1.2241 };
	arma::Mat<double> jacobi;

	// Option 1:
	auto F = make_function<double>(F_lmda, G_lmda);
	autodiff(F(x_val, x_val + 2));
	jacobian(jacobi, F);
	jacobi.print("Jacobian");

	// Option 2:
	jacobian<double>(jacobi, x_val, x_val + 2, F_lmda, G_lmda); // variadic in last argument
	jacobi.print("Jacobian");
}

// Hessian
void hessian()
{
	using namespace ad;
	auto F_lmda = MAKE_LMDA(
		x[0] * ad::sin(x[1]),
		w[0] + x[0] * x[1],
		ad::exp(w[1] * w[0])
	);
	double x_val[] = { -0.201, 1.2241 };
	arma::Mat<double> hess;
	arma::Mat<double> jacobi;

	// Computes Hessian and stores into "hess"
	hessian(hess, F_lmda, x_val, x_val + 2);
	// Computes Hessian and stores Hessian into "hess" and Jacobian into "jacobi"
	hessian(hess, jacobi, F_lmda, x_val, x_val + 2);

	hess.print("Hessian");
	jacobi.print("Jacobian");
}

int main()
{
	forward();
	reverse_simple();
	reverse_vec();
	reverse_function();
	reverse_jacobian();
	reverse_jacobian_2();
	reverse_vector();
	hessian();
	return 0;
}
