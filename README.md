# FastAD

## Description

FastAD is a C++ implementation of automatic differentiation providing both forward and reverse mode.
Reverse mode is based on expression template design and template metaprogramming.
Forward mode is computed without any expression templates.
FastAD was developed in Visual Studio 2017 using C++14 standard.

## Tutorial

The following is a simple use-case of FastAD.
**Var<T>** is variable containing datatype **T**.
In this example, **expr** is an expression and autodiff evaluates this expression to compute the gradient.
The gradient is stored in **Var<T>** as **df**.

```cpp
#include <autodiff.hpp>
#include <iostream>

int main() 
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
```

We may use **Vec<T>**, vector of **Var<T>** if the number of variables get longer.
The same code using vector is shown below.

```cpp
int main() 
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
```

If the access to **Vec<T>** is not needed and user wishes to encapsulate it, user may use Function object.
Function object requires a special form of lambda function shown below.
The following code uses Function object to perform the same job as before.

```cpp
int main()
{
	using namespace ad;
	double x_val[] = { -0.201, 1.2241 };
	auto F_lmda = [](auto& x, auto& w) {
		return std::make_tuple(
			x[0] * ad::sin(x[1]),
			w[0] + x[0] * x[1],
			ad::exp(w[1] * w[0])
		);
	};
	auto F = make_function<double>(F_lmda); // makes Function object with numeric computation type double
	autodiff(F(x_val, x_val + 2));			// F(begin, end) creates an expression as before

	std::cout << F.x[0].df << std::endl;
	std::cout << F.x[1].df << std::endl;
}
```

Finally, one may further encapsulate this procedure with **MAKE_LMDA** that creates the lambda function in the desired form.
The following uses the matrix library *armadillo* to store the Jacobian.
Note that any library supporting 2D-array (matrix) is viable subject to certain properties (documentation).
We may store the Jacobian after performing **autodiff** by passing the matrix object and Function object to **ad::jacobian**.
Note that this overload is variadic in last argument so multiple lambda functions can be passed.

```cpp
#include <armadillo>

int main()
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

	return 0;
}
```

Finally, one may encapsulate further by calling an overload of **ad::jacobian**.
We pass the numeric computation type (e.g. double) as template parameter, the 2D-array object, lambda function, begin and end iterators to actual x-values.

```cpp
#include <armadillo>

int main()
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
	jacobian<double>(jacobi, F_lmda, x_val, x_val + 2);
	jacobi.print("Jacobian");					// armadillo feature

	return 0;
}
```

## Author
- James Yang
