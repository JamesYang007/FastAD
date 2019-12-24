# FastAD 

[![Build Status](https://travis-ci.org/JamesYang007/FastAD.svg?branch=master)](https://travis-ci.org/JamesYang007/FastAD) 
[![Coverage Status](https://coveralls.io/repos/github/JamesYang007/FastAD/badge.svg?branch=master)](https://coveralls.io/github/JamesYang007/FastAD?branch=master)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/5fe0893b770643e7bd9d4c9ad6ab189b)](https://www.codacy.com/manual/JamesYang007/FastAD?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=JamesYang007/FastAD&amp;utm_campaign=Badge_Grade)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/3532/badge)](https://bestpractices.coreinfrastructure.org/projects/3532)
[![GitHub tag (latest by date)](https://img.shields.io/github/v/tag/JamesYang007/FastAD)](https://github.com/JamesYang007/FastAD/tags)
[![GitHub release (latest by date)](https://img.shields.io/github/v/release/JamesYang007/FastAD?include_prereleases)](https://github.com/JamesYang007/FastAD/releases)
[![GitHub issue (latest by date)](https://img.shields.io/github/issues/JamesYang007/FastAD)](https://github.com/JamesYang007/FastAD/issues)
[![License](https://img.shields.io/github/license/JamesYang007/FastAD?color=blue)](http://badges.mit-license.org)

## Table of contents

- [Overview](#overview)
  - [Intuitive Syntax](#intuitive-syntax)
  - [Simplicity](#simplicity)
  - [Robustness](#robustness)
  - [Memory Efficiency](#memory-efficiency)
  - [Speed](#speed)
  - [Features](#features)
- [Installation](#installation)
  - [Setup](#setup)
  - [Build and Install](#build-and-install)
- [Integration](#integration)
  - [CMake](#cmake)
- [Tests and Benchmarks](#tests-and-benchmarks)
  - [Linux and MacOS](#linux-and-macos)
- [User Guide](#user-guide)
  - [Forward Mode](#forward-mode)
  - [Reverse Mode](#reverse-mode)
    - [Basic](#basic)
    - [Using Vec](#using-vec)
    - [Using Expression Generator](#using-expression-generator)
    - [Jacobian](#jacobian)
    - [Hessian](#hessian)
- [Contact](#contact)
- [Team](#team)
- [Third Party Tools](#third-party-tools)
- [License](#license)

## Overview

FastAD is a header-only C++ template library of automatic differentiation supporting both forward and reverse mode to compute gradient and hessian. 
It utilizes the latest features in C++17 and expression templates for efficient computation.
FastAD is unique for the following:

### Intuitive syntax

Syntax choice is very important for C++ developers. 
Our philosophy is that syntax should be as similar as possible to STL.
All elementary functions such as `sin`, `exp`, and `pow` preserve the same name as those in STL.
This provides a seemless integration of FastAD in other existing projects as well with minor syntactical changes.

### Simplicity

FastAD is incredibly easy to use, partly due to the intuitive syntax.
With only a few lines of code, users can differentiate any smooth function.
See [User Guide](#user-guide) for more information.

### Robustness

FastAD has been heavily unit-tested with high test coverage followed by a few integration tests.
A variety of functions from simple (unit-test) to complex (integration-test) 
have been tested against manually-computed solutions.
At machine-level precision, the derivatives coincide.

### Memory Efficiency

FastAD is written to be incredibly efficient with memory usage and cache hits.
The main overhead of most AD libraries is the tape, which stores adjoints.
Using expression template techniques, we can significantly reduce this overhead.

### Speed

Speed is the utmost critical aspect of any AD library.
FastAD has been proven to be extremely fast, which inspired the name of this library.
Benchmark shows over 40-50x improvement from [Adept](https://github.com/rjhogan/Adept-2), an existing AD library.
Moreover, it also shows 10x improvement from the naive (and often inaccurate) finite-difference method.

### Features

- Forward and reverse mode automatic differentiation
- Easy way to compute and store jacobian and hessian
- Supports differentiation of polynomials and most elementary functions in standard library
- Follows STL syntax and semantics

## Installation

### Setup

- GCC >= 5
- Clang >= 5
- CMake >= 3.9

Optionally, user can install [Adept](https://github.com/rjhogan/Adept-2) if they wish to run benchmarks against it.

### Build and Install

Run the following command:

```shell
git clone --recurse-submodules https://github.com/JamesYang007/FastAD.git ~/FastAD
cd ~/FastAD
./setup.sh
./install.sh
```

## Integration

### CMake

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

## Tests and Benchmarks

### Linux and MacOS

To run tests, execute the following:

```shell
cd build/release && ctest
```

To run benchmarks, change directory to `build/release/benchmark` and run any one of the executables.
The following is an example command:

```shell
cd build/release/benchmark && ./ad_benchmark
```

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

## Contact

If you have any questions about FastAD, please [open an issue](https://github.com/JamesYang007/FastAD/issues/new).
When opening an issue, please describe in the fullest detail with a minimal example to recreate the problem.

For other general questions that cannot be resolved through opening issues,
feel free to [send me an email](mailto:jamesyang916@gmail.com).

## Team

| **James Yang** | **Kent Hall** |
| :---: | :---: |
| [![JamesYang007](https://avatars3.githubusercontent.com/u/5008832?s=100&v=4)](https://github.com/JamesYang007) | [](https://github.com/kentjhall) |
| <a href="http://github.com/JamesYang007" target="_blank">`github.com/JamesYang007`</a> | <a href="http://github.com/kentjhall" target="_blank">`github.com/kentjhall`</a> |

## Third Party Tools

Many third party tools were used for this project.

- [Clang](https://clang.llvm.org/): main compiler used for development.
- [Codacy](https://app.codacy.com/welcome/organizations): rigorous code analysis.
- [Github Changelog Generator](https://github.com/github-changelog-generator/github-changelog-generator): generate [CHANGELOG](https://github.com/JamesYang007/FastAD/blob/master/CHANGELOG.md).
- [Google Benchmark](https://github.com/google/benchmark): benchmark against various methods.
- [Googletest](https://github.com/google/googletest): unit-test and integration-test.
- [Travis](https://travis-ci.org/): continuous integration for Linux.
- [Valgrind](http://valgrind.org/): check memory leak/error.

## License

- **[MIT license](http://opensource.org/licenses/mit-license.php)**
- Copyright 2019 ©JamesYang007.
