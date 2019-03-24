#include <autodiff.hpp>
#include <armadillo>

void demo1()
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

void demo2()
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

void demo3()
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

void demo4()
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

void demo5()
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


int main()
{
	demo1();
	demo2();
	demo3();
	demo4();
	demo5();
	return 0;
}
