# FastAD

## Description
FastAD is a C++ implementation of automatic differentiation providing both forward and reverse mode.
Reverse mode is based on expression template design and template metaprogramming.
Forward mode is computed without any expression templates.
FastAD was developed in Visual Studio 2017 using C++14 standard.

## Demo Code
The following is a self-contained demo code using the matrix library *armadillo* to store the Jacobian.
Note that any library supporting 2D-array (matrix) is viable subject to certain properties (documentation).
We wish to compute the gradient of 
f(x_0, x_1, x_2, x_3, x_4) = sin(x_0)cos(x_1) + x_2 + x_3x_4
evaluated at (0.1, 2.3, -1., 4.1, -5.21).

```cpp
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
```

We create a lambda function using the macro *MAKE_LMDA* and store the jacobian in *jacobi* evaluated at the given points of *x*.
We then invoke a print method defined for armadillo matrices.

## Author
- James Yang
