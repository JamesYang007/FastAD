# FastAD [![Build Status](https://travis-ci.org/JamesYang007/FastAD.svg?branch=master)](https://travis-ci.org/JamesYang007/FastAD) ![GitHub tag (latest by date)](https://img.shields.io/github/v/tag/JamesYang007/FastAD) ![GitHub release (latest by date)](https://img.shields.io/github/v/release/JamesYang007/FastAD?include_prereleases) [![License](http://img.shields.io/:license-mit-blue.svg?style=flat-square)](http://badges.mit-license.org)

FastAD is a C++ template library of automatic differentiation supporting both forward and reverse mode to compute gradient and hessian. 
It utilizes the latest features in C++17 and expression templates for efficient computation.
---

## Table of contents
* [Installation](#installation)
  * [Setup](#setup)
  * [Build and Install](#build-and-install)
* [Features](#features)
* [Tests and Benchmarks](#tests-and-benchmarks)
  * [Linux/MacOS](#linux/macos)
* [User Guide](#user-guide)
  * [Include](#include)
  * [Forward Mode](#forward-mode)
  * [Reverse Mode](#reverse-mode)
    * [Basic](#basic)
    * [Using Vec](#using-vec)
    * [Using Expression Generator](#using-expression-generator)
    * [Jacobian](#jacobian)
    * [Hessian](#hessian)
* [Team](#team)
- [FAQ](#faq)
  - [How do I build my project with FastAD as a dependency?](#how-do-i-build-my-project-with-fastad-as-a-dependency)
* [Support](#support)
* [License](#license)

---

## Installation

### Setup

* Install a compiler with full support for C++17 standard.
* (Linux/MacOS) Install cmake >= 3.9
* (optional) [Adept](https://github.com/rjhogan/Adept-2) if user wishes to run benchmarks against it as well

### Build and Install

Run the following command:

```shell
git clone --recurse-submodules https://github.com/JamesYang007/FastAD.git ~/FastAD
cd ~/FastAD
./setup.sh
./install.sh
```

---

## Features

* Forward-mode and reverse-mode automatic differentiation
* Easy way to compute and store jacobian and hessian
* Supports differentiation of polynomials and most elementary functions in standard library
* Follows STL syntax and semantics for accessibility

---

## Tests and Benchmarks

### Linux/MacOS

To run tests, execute the following:

```shell
cd build/release && ctest
```

To run benchmarks, change directory to `build/release/benchmark` and run any one of the executables
The following is an example command:

```shell
cd build/release/benchmark && ./ad_benchmark
```

---

## User Guide

### Forward Mode

```cpp
#include <fastad>

int main()
{
    using namespace ad;

    ForwardVar<double> w1(0.), w2(1.);
    w1.set_adjoint(1.);
    ForwardVar<double> w3 = w1 * sin(w2);
    ForwardVar<double> w4 = w3 + w1 * w2;
    ForwardVar<double> w5 = exp(w4 * w3);

    std::cout << "f(x, y) = exp((x * sin(y) + x * y) * x * sin(y))\n"
              << "df/dx = " << w5.get_adjoint() << std::endl;
    return 0;
}
```

`ForwardVar` is class template whose template parameter is the type of the underlying values.
We initialize `w1` and `w2` with values `0.` and `1.`, respectively, and set the adjoint of `w1` to `1.`.
By default, all adjoints of `ForwardVar` are set to `0.`.
This indicates that we will be differentiating in the direction of `(1.,0.)`, i.e. partial derivative w.r.t. `w1`.
After computing the desired expression, we get the directional derivative by calling `get_adjoint()` on the final `ForwardVar` object.

### Reverse Mode

#### Basic

```cpp
#include <fastad>

int main()
{
    using namespace ad;

    Var<double> w1(0.), w2(1.), w3, w4, w5;
    auto expr = (w3 = w1 * sin(w2), 
                 w4 = w3 + w1 * w2, 
                 w5 = exp(w4 * w3));
    autodiff(expr);

    std::cout << "f(x, y) = exp((x * sin(y) + x * y) * x * sin(y))\n"
              << "df/dx = " << w1.get_adjoint() << std::endl
              << "df/dy = " << w2.get_adjoint() << std::endl;
    return 0;
}
```

`Var` is the reverse-mode variable that will store all partial derivatives of the final expression.
We may use `Var` as a placeholder in a bigger expression as shown in the definition of `expr`.
Placeholder equations can be "glued" by using the comma-operator to form one big expression.
Note that `expr` is not necessarily of type `Var<double>`.
To differentiate, simply call `autodiff` on the final expression `expr`.
To get the adjoints, simply call `get_adjoint()` for the initial variables `w1, w2`.

#### Using Vec

```cpp
#include <fastad>

int main()
{
    using namespace ad;

    Vec<double> x({0., 1.});
    Vec<double> w(3);
    auto expr = (w[0] = x[0] * sin(x[1]), 
                 w[1] = w[0] + x[0] * x[1], 
                 w[2] = exp(w[1] * w[0]));
    autodiff(expr);

    std::cout << "f(x, y) = exp((x * sin(y) + x * y) * x * sin(y))\n"
              << "df/dx = " << x[0].get_adjoint() << std::endl
              << "df/dy = " << x[1].get_adjoint() << std::endl;
    return 0;
}
```

`Vec` is a container of `Var`.
It can be initialized using an initializer list where every value is used to construct a `Var`.
The expression can be constructed similarly by accessing the appropriate elements of a `Vec` object.

#### Using Expression Generator
Using the same `x` variable from previous example, one may construct `expr` similarly, 
but without defining the temporary placeholders `w`.

```cpp
auto F_lmda = [](const auto& x, const auto& w) {
    return (w[0] = x[0] * sin(x[1]),
            w[1] = w[0] + x[0] * x[1],
            w[2] = exp(w[1] * w[0]));
};
auto gen = make_exgen<double>(F_lmda);
auto expr = gen.generate(x);
```

An expression generator returned by `make_exgen` encapsulates the temporary placeholders `w` shown in the previous examples.
This object will not create an expression until user calls `generate(x)` where `x` is the initial `Vec` object.
The template parameter to `make_exgen` is the type of the values and will usually be identical to the template parameter of the `Vec` object `x`.

#### Jacobian
Using the same `F_lmda` lambda function from previous example, 
one can further encapsulate this process of differentiating by only supplying the raw x-values
and a lambda function (such as `F_lmda`) that will generate the expression.

```cpp
double x_val[] = {0., 1.};
Mat<double> jacobi;	
jacobian(jacobi, x_val, x_val + 2, F_lmda);
std::cout << "f(x, y) = exp((x * sin(y) + x * y) * x * sin(y))" << std::endl;
jacobi.print("Jacobian of f(x, y)");
```

#### Hessian
Using the same `x_val` and `F_lmda` from previous example,
we may compute the hessian in the following way:

```cpp
Mat<double> hess;
hessian(hess, x_val, x_val + 2, F_lmda);
std::cout << "f(x, y) = exp((x * sin(y) + x * y) * x * sin(y))" << std::endl;
hess.print("Hessian of f(x, y)");
```

---

## Team

| **James Yang** | **Kent Hall** |
| :---: | :---: |
| [![JamesYang007](https://avatars3.githubusercontent.com/u/5008832?s=100&v=4)](https://github.com/JamesYang007) | [](https://github.com/kentjhall) |
| <a href="http://github.com/JamesYang007" target="_blank">`github.com/JamesYang007`</a> | <a href="http://github.com/kentjhall" target="_blank">`github.com/kentjhall`</a> |

---

## FAQ

### How do I build my project with FastAD as a dependency?
If project is built using CMake, add the following to CMakeLists.txt in the root directory:

```cmake
find_package(FastAD CONFIG REQUIRED)
```

and use `target_link_libraries` to link with `FastAD::FastAD`.

An example project that uses FastAD as a dependency may have a CMakeLists.txt that looks like this:

```cmake
project("MyProject")
find_package(FastAD CONFIG REQUIRED)
add_executable(main src/main.cpp)
target_link_libraries(main FastAD::FastAD)
```

---

## Support
Feel free to contact me via:
* Email: jamesyang916@gmail.com

---

## License

[![License](http://img.shields.io/:license-mit-blue.svg?style=flat-square)](http://badges.mit-license.org)

- **[MIT license](http://opensource.org/licenses/mit-license.php)**
- Copyright 2019 ©JamesYang007.
