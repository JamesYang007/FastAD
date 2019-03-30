# FastAD

## Description

FastAD is a C++ implementation of automatic differentiation providing both forward and reverse mode.
Reverse mode is based on expression template design and template metaprogramming.
Forward mode is computed without any expression templates.
FastAD was developed in Visual Studio 2017 using C++14 standard.

## Tutorial

### Forward AD

Forward AD only requires the use of **ForwardVar<T>**.
Each **ForwardVar<T>** variable contains the x-value and directional derivative up to that variable.
In the following example, **w1.df = 1** and by default all other **df** member variables are **0**.
This will compute the partial derivative with respect to **w1**.
There are no expressions used for **ForwardVar<T>**, and each computation will return another **ForwardVar<T>**.

```cpp
#include <autodiff.hpp>

int main()
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

	return 0;
}
```

### Reverse AD

#### Scalar Function

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

If the access to **Vec<T>** is not needed and user wishes to encapsulate it, user may use a Function object.
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
	arma::Mat<double> jacobi;					// substitute for a 2D-array data structure
												// satisfying certain properties
												// (more information in documentation)
	jacobian<double>(jacobi, F_lmda, x_val, x_val + 2);
	jacobi.print("Jacobian");					// armadillo feature

	return 0;
}
```

#### Vector-Valued Function

With vector-valued function, a tuple of scalar functions, one can manually perform the task for scalar functions for each scalar function of a vector-valued function.
Using Function Object, the syntax is almost the same as before.

```cpp
int main()
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

	// Option 1: Function object
	auto F = make_function<double>(F_lmda, G_lmda);
	autodiff(F(x_val, x_val + 2));
	jacobian(jacobi, F);
	jacobi.print("Jacobian");

	// Option 2: ad::jacobian<type>
	jacobian<double>(jacobi, x_val, x_val + 2, F_lmda, G_lmda); // variadic in last argument
	jacobi.print("Jacobian");
	return 0;
}
```

### Hessian

We can also compute the Hessian of a scalar function (only).
The **ad::hessian** function computes the Hessian and stores it into a 2D-array.
The computation for Hessian requires the computation of the gradient, 
hence the gradient will be computed and can also be retrieved by passing another 2D-array.

```cpp
int main()
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

	return 0;
}
```

## Further Implementation

- Think about Delta Function

## Author
- James Yang
