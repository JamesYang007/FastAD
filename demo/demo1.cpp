#include <autodiff.hpp>
#include <armadillo>

int main()
{
	using namespace ad;
	auto&& F_lmda = MAKE_LMDA(
		ad::sin(x[0])*ad::cos(x[1]),
		x[2] + x[3] * x[4],
		w[0] + w[1]
	);
	double x[] = { 0.1, 2.3, -1., 4.1, -5.21 }; // substitute for any data structure that is iterable
	arma::Mat<double> jacobi;					// substitute for a 2D-array data structure
												// satisfying certain properties
												// (more information in documentation)
	jacobian<double>(jacobi, x, x + 5, F_lmda);
	jacobi.print("Jacobian");					// armadillo feature

	return 0;
}