# FastAD

## Description
FastAD is a C++ implementation of automatic differentiation providing both forward and reverse mode.
Reverse mode is based on expression template design and template metaprogramming.
Forward mode is computed without any expression templates.
FastAD was developed in Visual Studio 2017 using C++14 standard.

## Demo Code
The following is a demo code using the matrix library *armadillo* to store the Jacobian.
We create a lambda function using the macro *MAKE_LMDA* and store the jacobian in *jacobi* evaluated at the given points of *x*.

```cpp
auto&& F_lmda = MAKE_LMDA(
	ad::sin(x[0])*ad::cos(x[1]),
	x[2] + x[3] * x[4],
	w[0] + w[1]
);
double x[] = { 0.1, 2.3, -1., 4.1, -5.21 };
arma::Mat<double> jacobi;
ad::jacobian<double>(jacobi, x, x + 5, F_lmda);
```

## Author
- James Yang
