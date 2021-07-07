# FastAD 

[![Build Status](https://travis-ci.org/JamesYang007/FastAD.svg?branch=master)](https://travis-ci.org/JamesYang007/FastAD) 
[![CircleCI](https://circleci.com/gh/JamesYang007/FastAD/tree/master.svg?style=svg)](https://circleci.com/gh/JamesYang007/FastAD/tree/master)
[![Coverage Status](https://coveralls.io/repos/github/JamesYang007/FastAD/badge.svg?branch=master&service=github)](https://coveralls.io/github/JamesYang007/FastAD?branch=master)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/5fe0893b770643e7bd9d4c9ad6ab189b)](https://www.codacy.com/manual/JamesYang007/FastAD?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=JamesYang007/FastAD&amp;utm_campaign=Badge_Grade)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/3532/badge)](https://bestpractices.coreinfrastructure.org/projects/3532)
[![GitHub tag (latest by date)](https://img.shields.io/github/v/tag/JamesYang007/FastAD)](https://github.com/JamesYang007/FastAD/tags)
[![GitHub release (latest by date)](https://img.shields.io/github/v/release/JamesYang007/FastAD?include_prereleases)](https://github.com/JamesYang007/FastAD/releases)
[![GitHub issue (latest by date)](https://img.shields.io/github/issues/JamesYang007/FastAD)](https://github.com/JamesYang007/FastAD/issues)
[![License](https://img.shields.io/github/license/JamesYang007/FastAD?color=blue)](http://badges.mit-license.org)

## Table of contents

- [Overview](#overview)
- [Installation](#installation)
  - [General Users](#general-users)
  - [Developers](#developers)
- [Integration](#integration)
  - [CMake](#cmake)
  - [Others](#others)
- [User Guide](#user-guide)
  - [Forward Mode](#forward-mode)
  - [Reverse Mode](#reverse-mode)
    - [Basic Usage](#basic-usage)
    - [Placeholder](#placeholder)
    - [Advanced Usage](#advanced-usage)
- [Applications](#applications)
  - [Black-Scholes Put-Call Option Pricing](#black-scholes-put-call-option-pricing)
- [Quick Reference](#quick-reference)
  - [Forward](#forward)
  - [Reverse](#reverse)
- [Contact](#contact)
- [Contributors](#contributors)
- [Third Party Tools](#third-party-tools)
- [License](#license)

## Overview

FastAD is a header-only C++ template library for automatic differentiation supporting both forward and reverse mode. 
It utilizes the latest features in C++17 and expression templates for efficient computation.
FastAD is unique for the following:

### Intuitive syntax

Syntax choice is very important for C++ developers. 
Our philosophy is that syntax should be as similar as possible to mathematical notation.
This makes FastAD easy to use and allow users to write readable, intuitive, and simple code.
See [User Guide](#user-guide) for more details.

### Robustness

FastAD has been heavily unit-tested with high test coverage followed by a few integration tests.
A variety of functions have been tested against analytical solutions;
at machine-level precision, the derivatives coincide.

### Memory Efficiency

FastAD is written to be incredibly efficient with memory usage and cache hits.
The main overhead of most AD libraries is the tape, which stores adjoints.
Using expression template techniques, and smarter memory management,
we can significantly reduce this overhead.

### Speed

Speed is the utmost critical aspect of any AD library.
FastAD has been proven to be extremely fast, which inspired the name of this library.
Benchmark shows over orders of magnitude improvement from existing libraries 
such as Adept, Stan Math Library, ADOL-C, CppAD, and Sacado
(see our separate benchmark repositor [ADBenchmark](https://github.com/JamesYang007/ADBenchmark)).
Moreover, it also shows 10x improvement from the naive (and often inaccurate) finite-difference method.

## Installation

First, clone the repo:
```
git clone https://github.com/JamesYang007/FastAD.git ~/FastAD
```

From here on, we will refer to the cloned directory as `workspace_dir`
(in the example above, `workspace_dir` is `~/FastAD`).

The library has the following dependencies:
- Eigen3.3
- GoogleTest (dev only)
- Google Benchmark (dev only)

### General Users

If the user already has Eigen3.3 installed in their system, they can omit the following step.
For general users, if they wish to install Eigen locally, they can run 
```bash
./setup.sh
``` 
from `workspace_dir`. 
This will install Eigen3.3 into `workspace_dir/libs/eigen-3.3.7/build`.

For those who want to install `FastAD` globally into the system, simply run:
```bash
./install.sh
```
This will build and install the header files into the system.

For users who want to install `FastAD` locally, run the following from `workspace_dir`:
```bash
mkdir -p build && cd build
cmake -DCMAKE_INSTALL_PREFIX=. -DFASTAD_ENABLE_TEST=OFF ..
make install
```
One can set the `CMAKE_INSTALL_PREFIX` to anything.
This example will install the library in `workspace_dir/build`.

### Developers

Run the following to install all of the dependencies locally:
```bash
./setup.sh dev
```

To build the library, run the following:
```bash
./clean-build.sh <debug/release> [other CMake flags...]
```
Here are the following options one can specify as a CMake flag `-D...=ON`
(replace `...` with any of the following):
- FASTAD_ENABLE_TEST        (builds tests)
- FASTAD_ENABLE_BENCHMARK   (builds benchmarks)
- FASTAD_ENABLE_EXAMPLE     (builds examples)

By default, with the exception of `FASTAD_ENABLE_TEST`, the flags are `OFF`.
Note that this only builds and does not install the library.

To run tests, execute the following:
```bash
cd build/<debug/release>
ctest -j6
```

To run benchmarks, change directory to 
`build/<debug/release>/benchmark` and run any one of the executables.

## Integration

### CMake

If your project is built using CMake, add the following to CMakeLists.txt in the root directory:
```cmake
find_package(FastAD CONFIG REQUIRED)
```

If you installed the library locally, say `path_to_install`, then add the following:
```cmake
find_package(FastAD CONFIG REQUIRED HINTS path_to_install/share)
```

For any program that requires `FastAD`, 
use `target_link_libraries` to link with `FastAD::FastAD`.

An example project that uses FastAD as a dependency may have a CMakeLists.txt that looks like this:
```cmake
project("MyProject")
find_package(FastAD CONFIG REQUIRED)
add_executable(main src/main.cpp)
target_link_libraries(main FastAD::FastAD)
```

### Others

Simply add the following flag when compiling your program:
```
-Ipath_to_install/include
```

An example build command would be:
```
g++ main.cpp -Ipath_to_install/include
```

If Eigen3.3 was installed locally, you must provide its path as well.

## User Guide

The only header the user needs to include is `fastad`.

### Forward Mode

Forward mode is extremely simple to use.

The only class a user will need to deal with is `ForwardVar<T>`,
where `T` is the underlying data type (usually `double`).
The API only exposes getters and setters:
```cpp
ForwardVar<double> v;       // initialize value and adjoint to 0
v.set_value(1.);            // value is now 1.
double r = v.get_value();   // r is now 1.
v.set_adjoint(1.);          // adjoint is now 1.
double s = v.get_adjoint(); // s is now 1.
```

The rest of the work has already been done by the library
with operator overloading.

Here is an example program that differentiates a complicated function:
```cpp
#include <fastad>
#include <iostream>

int main()
{
    using namespace ad;

    ForwardVar<double> w1(0.), w2(1.);
    w1.set_adjoint(1.); // differentiate w.r.t. w1
    ForwardVar<double> w3 = w1 * sin(w2);
    ForwardVar<double> w4 = w3 + w1 * w2;
    ForwardVar<double> w5 = exp(w4 * w3);

    std::cout << "f(x, y) = exp((x * sin(y) + x * y) * x * sin(y))\n"
              << "df/dx = " << w5.get_adjoint() << std::endl;
    return 0;
}
```

We initialize `w1` and `w2` with values `0.` and `1.`, respectively.
We set the adjoint of `w1` to `1.` to indicate that we are differentiating w.r.t. `w1`.
By default, all adjoints of `ForwardVar` are set to `0.`.
This indicates that we will be differentiating in the direction of `(1.,0.)`, i.e. partial derivative w.r.t. `w1`.
Note that user could also set the adjoint for `w2`, 
__but this will compute the directional derivative multiplied by the norm of (w1, w2)__.
After computing the desired expression, we get the directional derivative 
by calling `get_adjoint()` on the final `ForwardVar` object.

### Reverse Mode

#### Basic Usage

The most basic usage simply requires users to create `Var<T, ShapeType>` objects.
`T` denotes the underlying value type (usually `double`).
`ShapeType` denotes the general shape of the variable.
It must be one of `ad::scl, ad::vec, ad::mat` corresponding to
scalar, (column) vector, and matrix, respectively.

```cpp
Var<double, scl> x;
Var<double, vec> v(5);    // set size to 5
Var<double, mat> m(2, 3); // set shape to 2x3
```

From here, one can create complicated expressions 
by invoking a wide range of functions 
(see [Quick Reference](#quick-reference) for a full list of expression builders).

As an example, here is an expression to differentiate `sin(x) + cos(v)`:
```cpp
auto expr = (sin(x) + cos(v));
```
Note that this represents a vector expression, since `sin(x)` is a scalar expression
but `cos(v)` is a vectorized function on a vector, which is again a vector expression.

Before we differentiate, the expression is required to
"bind" to a storage for the values and adjoints of intermediate expression nodes.
The reason for this design is for speed purposes and cache hits.
If the user wishes to manage this storage, they can do this:
```cpp
auto size_pack = expr.bind_cache_size();
std::vector<double> val_buf(size_pack(0));
std::vector<double> adj_buf(size_pack(1));
expr.bind_cache({val_buf.data(), adj_buf.data()});
```

The `bind_cache_size()` will return exactly how many doubles are needed
for values and adjoints, respectively, of type `util::SizePack`, which is an alias for `Eigen::Array<size_t, 2, 1>`.
and `bind(util::PtrPack<double>)` will bind itself to that region of memory.
It is encouraged to create the pointer pack object using initializer list as shown above.
This pattern occurs so often that if the user does not care about managing this,
they should use the following helper function:
```cpp
auto expr_bound = ad::bind(sin(x) + cos(v));
```

`ad::bind` will return a wrapper class that wraps the expression
and at construction binds it to a privately owned storage 
in the same way described above.

_If the expression is not bound to any storage, it will lead to segfault_!

To differentiate the expression, simply call the following:
```cpp
auto f = ad::autodiff(expr_bound, seed);

// or if the raw expression is manually bound,
auto f = ad::autodiff(expr, seed);
```
where `seed` is the initial adjoint for the root of the expression.
If the expression is scalar, seed is a literal (`double`)
and the default value is `1`, so the user does not have to input anything.
If the expression is multi-dimensional, 
seed does not have a default value,
must be of type `Eigen::Array`,
and must have the same dimensions as the expression.
`autodiff` will return the evaluated function value.
This return value is `T` if it is a scalar expression, and otherwise,
`Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, ...>>` where `...` depends on
the shape of the expression (`1` if vector, `Eigen::Dynamic` if matrix).

You can retrieve the adjoints by calling `get_adj(i,j)` or 
`get_adj()` (with no arguments) from `x, v` like so:
```cpp
x.get_adj(0,0); // (1) get adjoint for x 
v.get_adj(2,0); // (2) get adjoint for v at index 2
x.get_adj();    // (3) get full adjoint (same as (1))
v.get_adj();    // (4) get full adjoint
```

The full code for this example is the following:
```cpp
#include <fastad>
#include <iostream>

int main()
{
    using namespace ad;

    Var<double, scl> x(2);
    Var<double, vec> v(5);

    auto expr_bound = bind(sin(x) + cos(v));
    auto f = autodiff(expr_bound);

    std::cout << x.get_adj() << std::endl;
    std::cout << v.get_adj(2,0) << std::endl;

    return 0;
}
```

_Note: once you have differentiated an expression, 
you must reset the adjoints of all variables to 0 before differentiating again.
This includes placeholder variables (see below)._
To that end, we provide a member function for `Var` called `reset_adj()`.

Here is a more complicated example:

```cpp
#include <fastad>

int main()
{
    Var<double, vec> v1(6);
    Var<double, vec> v2(5);
    Var<double, mat> M(5, 6);
    Var<double, vec> w(5);
    Var<double, scl> r;

    auto& v1_raw = v1.get();  // Eigen::Map
    auto& v2_raw = v2.get();  // Eigen::Map
    auto& M_raw = M.get();  // Eigen::Map

    // initialize...

    auto expr = bind((
        w = ad::dot(M, v1) + v2,
        r = sum(w) * sum(w * v2)
    ));

    autodiff(expr);

    std::cout << v1.get_adj(0,0) << std::endl;  // adjoint of v1 at index 0
    std::cout << v2.get_adj(1,0) << std::endl;  // adjoint of v2 at index 1
    std::cout << M.get_adj(1,2) << std::endl;   // adjoint of M at index (1,2)

    return 0;
}
```

#### Placeholder

In the previous example, we used a placeholder expression,
which is of the form `v = expr`.
You can use placeholders to greatly speed up the performance
and also save a lot of memory.

Consider the following expression:
```cpp
auto expr = (sin(x) + cos(v) + sum(cos(v)));
```
When there are common expressions (like `cos(v)`),
they will be evaluated multiple times unnecessarily.

Placeholder expressions are created by using `operator=` with
a `Var` and an expression:
```cpp
Var<double, scl> x;
Var<double, vec> v(5);
Var<double, vec> w(v.size());
auto expr = (
    w = cos(v),
    sin(x) + w + sum(w)
);
```
This will only evaluate `cos(v)` once, and reuse the results
for the subsequent expressions by using `w`.

While this is not specific to placeholder expressions,
`operator,` is usually invoked to "glue" many placeholder expressions.
However, one can certainly glue any kinds of expressions, if they wish.

#### Advanced Usage

For advanced users who need to get more low-level control over 
the memory for values and adjoints for all variables, they can use
`VarView<T, ShapeType>`.
All of the discussion above holds for `VarView` objects.
In fact, when we build an expression out of `Var` of `VarView`,
we convert all of them to `VarView`s so that the expression is solely a viewer.

`VarView` objects __do not__ own the values and adjoints, but views them.
Here is an example program that binds the viewers to a contiguous chunk of memory:
```cpp
VarView<double, scl> x;
VarView<double, vec> v(3);
VarView<double, vec> w(3);

std::vector<double> vals(x.size() + v.size());
std::vector<double> adjs(x.size() + v.size());
std::vector<double> w_vals(w.size());
std::vector<double> w_adjs(w.size());

// x binds to the first element of storages
double* val_next = x.bind(vals.data());
double* adj_next = x.bind_adj(adjs.data());

// v binds starting from 2nd element of storages
v.bind(val_next);
v.bind_adj(adj_next);

// bind placeholders to a separate storage region
w.bind(w_vals.data());
w.bind_adj(w_adjs.data());

auto expr = (
    w = cos(v),
    sin(x) + w + sum(w)
);
```

## Applications

### Black-Scholes Put-Call Option Pricing

The following is an example of computing deltas in Black-Scholes model using FastAD.
This example was taken from the autodiff library in 
[boost](https://www.boost.org/doc/libs/master/libs/math/doc/html/math_toolkit/autodiff.html#math_toolkit.autodiff.example-black_scholes).

```cpp
#include <fastad>
#include <iostream>

enum class option_type {
    call, put
};

// Standard Normal CDF
template <class T>
inline auto Phi(const T& x)
{
    return 0.5 * (ad::erf(x / std::sqrt(2.)) + 1.);
}

// Generates expression that computes Black-Scholes option price
template <option_type cp, class Price, class Cache>
auto black_scholes_option_price(const Price& S,
                                double K,
                                double sigma,
                                double tau,
                                double r,
                                Cache& cache)
{
    cache.resize(3);
    double PV = K * std::exp(-r * tau);
    auto common_expr = (
            cache[0] = ad::log(S / K),
            cache[1] = (cache[0] + ((r + sigma * sigma / 2.) * tau)) / 
                            (sigma * std::sqrt(tau)),
            cache[2] = cache[1] - (sigma * std::sqrt(tau))
    );
    if constexpr (cp == option_type::call) {
        return (common_expr,
                Phi(cache[1]) * S - Phi(cache[2]) * PV);
    } else {
        return (common_expr,
                Phi(-cache[2]) * PV - Phi(-cache[1]) * S);
    }
}

int main()
{
    double K = 100.0;        
    double sigma = 5;        
    double tau = 30.0 / 365; 
    double r = 1.25 / 100;   
    ad::Var<double> S(105);  
    std::vector<ad::Var<double>> cache;

    auto call_expr = ad::bind(
            black_scholes_option_price<option_type::call>(
                S, K, sigma, tau, r, cache));

    double call_price = ad::autodiff(call_expr);

    std::cout << call_price << std::endl;
    std::cout << S.get_adj() << std::endl;

    // reset adjoints before differentiating again
    S.reset_adj();
    for (auto& c : cache) c.reset_adj();

    auto put_expr = ad::bind(
            black_scholes_option_price<option_type::put>(
                S, K, sigma, tau, r, cache));

    double put_price = ad::autodiff(put_expr);

    std::cout << put_price << std::endl;
    std::cout << S.get_adj() << std::endl;

    return 0;
}
```

We observed the same output as the one shown in 
[boost](https://www.boost.org/doc/libs/master/libs/math/doc/html/math_toolkit/autodiff.html#math_toolkit.autodiff.example-black_scholes)
for the prices and deltas (S's adjoints).

### Quadratic Expression Differential

In ML applications, user usually provide a full parameter vector to an optimizer and write an objective function to calculate objective value and gradient. The gradient pointer is provided by the optimizer and user fill values. Then it will be more convinent to use `VarView` to bind gradient pointer as buffer.

Here is an example of differentiating a quadratic expression `x^T*Sigma*x` using `VarView`.

Note that currently there is no `transpose()` support so we crete two `VarView` but they share the same buffer.

```
#include "fastad"
#include <Eigen/src/Core/Matrix.h>
#include <iostream>

int main() {
    using namespace ad;

    // Generating buffer.
    Eigen::MatrixXd x_data(1, 2);
    x_data << 0.5, 0.6;
    Eigen::MatrixXd x_adj(1, 2);
    x_adj.setZero();

    // Initialize variable.
    VarView<double, mat> x_T(x_data.data(), x_adj.data(), 1, 2),
        x(x_data.data(), x_adj.data(), 2, 1);

    // Initialize matrix.
    Eigen::MatrixXd _Sigma(2, 2);
    _Sigma << 2, 3, 3, 6;
    std::cout << _Sigma << std::endl;
    auto Sigma = constant(_Sigma);

    // expression: x^T*Sigma*x
    auto quad_expr = bind(dot(dot(x_T, Sigma), x));
    Eigen::ArrayXd seed(1);
    seed.setOnes();
    auto f = autodiff(quad_expr, seed);

    // Print results.
    std::cout << "f: " << f << std::endl;
    std::cout << x_T.get() << std::endl;     //[0.5, 0.6]
    std::cout << x_T.get_adj() << std::endl; //[5.6, 10.2]

    return 0;
}
```

## Quick Reference

### Forward 

__ForwardVar<T>__:
- class representing a variable for forward AD
- `set_value(T x)`: sets value to x
- `get_value()`: gets underlying value
- `set_adjoint(T x)`: sets adjoint to x
- `get_adjoint()`: gets underlying adjoint

__Unary Functions__:
- unary minus: `operator-`
- trig functions: `sin, cos, tan, asin, acos, atan`
- others: `exp, log, sqrt`

__Operators__:
- binary: `+,-,*,/`
- increment: `+=`

### Reverse 

__Shape Types__:
- `ad::scl, ad::vec, ad::mat`

__VarView<T, ShapeType=scl>__:
- This is only useful for users who really want to optimize for performance
- `ShapeType` must be one of the types listed above
- `T` is the underlying value type
- `VarView(T* v, T* a, rows=1, cols=1)`:
    - constructs to view values starting from v,
      adjoints starting from a, and has the shape of rows x cols.
    - vector shapes must pass rows
    - matrix shapes must pass both rows and cols
- `VarView()`
    - constructs with nullptrs
- `.bind(T* begin)`: views values starting from begin
- `.bind_adj(T* begin)`: views adjoints starting from begin

__Var<T, ShapeType=scl>__:
- A `Var` is a `VarView` (views itself)
- Main difference with `VarView` is that it owns the values and adjoints
- Users will primarily use this class to represent AD variables.
- API is same as `VarView`

__Unary Functions (vectorized if multi-dimensional)__:
- unary minus: `operator-`
- trig functions: `sin, cos, tan, asin, acos, atan`
- others: `exp, log, sqrt, erf`

__Operators__:
- binary: `+,-,*,/`
- modification: `+=`, `-=`, `*=`, `/=`
- comparison: `<,<=,>,>=,==,!=,&&,||`
    - Note: `&&` and `||` are undefined behavior for 
      multi-dimensional non-boolean expressions
- placeholder: `operator=`
    - only overloaded for `VarView` expressions
- glue: `operator,`
    - any expressions can be "glued" using this operator
- unary minus: `operator-`
- trig functions: `sin, cos, tan, asin, acos, atan`
- others: `exp, log, sqrt`

__Special Expressions__:
- `ad::constant(T)`:
- `ad::constant(const Eigen::Vector<T, Eigen::Dynamic, 1>&)`:
- `ad::constant(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>&)`:
- `ad::constant_view(T*)`:
- `ad::constant_view(T*, rows)`:
- `ad::constant_view(T*, rows, cols)`:
- `ad::det<policy>(m)`:
    - determinant of matrix `m`
    - `policy` must be one of: `DetFullPivLU`, `DetLDLT`, `DetLLT`
        - see `Eigen` documentation for each of these (without the prefix `Det`) for 
          when they apply.
- `ad::dot(m, v)`:
    - represents matrix product with a matrix and a (column) vector
- `ad::for_each(begin, end, f)`:
    - generalization of operator,
    - represents evaluating expressions generated by `f` when fed with elements
      from `begin` to `end`.
- `ad::if_else(cond, if, else)`:
    - represents an if-else statement
    - `cond` MUST be a scalar expression
    - `if` and `else` must have the exact same shape
- `ad::log_det<policy>(m)`
    - same as `det<policy>(m)` but computes log-abs-determinant
    - `policy` must be one of: `LogDetFullPivLU`, `LogDetLDLT`, `LogDetLLT`
- `ad::norm(v)`:
    - represents the squared norm of a vector or Frobenius norm for matrix
- `ad::pow<n>(e)`:
    - compile-time known, integer-powered expression
- `ad::prod(begin, end, f)`:
    - represents the product of expressions generated by `f`
      when fed with elements from `begin` to `end`.
- `ad::prod(e)`:
    - represents the product of all _elements_ of the expression `e`
    - e.g. if `e` is a vector expression, it represents the product of all its elements.
- `ad::sum(begin, end, f)`:
- `ad::sum(e)`:
    - same as prod but represents summation

__Stats Expressions__:
All log-pdfs are adjusted to omit constants.
Parameters can have various combinations of shapes and follow the usual vectorized notion.
- `ad::bernoulli(x, p)`
- `ad::cauchy_adj_log_pdf(x, loc, scale)`
- `ad::normal_adj_log_pdf(x, mu, s)`
- `ad::uniform_adj_log_pdf(x, min, max)`
- `ad::wishart_adj_log_pdf(X, V, n)`

## Contact

If you have any questions about FastAD, please [open an issue](https://github.com/JamesYang007/FastAD/issues/new).
When opening an issue, please describe in the fullest detail with a minimal example to recreate the problem.

For other general questions that cannot be resolved through opening issues,
feel free to [send me an email](mailto:jamesyang916@gmail.com).

## Contributors

| **James Yang** | **Kent Hall** |
| :---: | :---: |
| [![JamesYang007](https://avatars3.githubusercontent.com/u/5008832?s=100&v=4)](https://github.com/JamesYang007) | [](https://github.com/kentjhall) |
| <a href="http://github.com/JamesYang007" target="_blank">`github.com/JamesYang007`</a> | <a href="http://github.com/kentjhall" target="_blank">`github.com/kentjhall`</a> |

## Third Party Tools

Many third party tools were used for this project.

- [Clang](https://clang.llvm.org/): main compiler used for development.
- [CMake](https://cmake.org/): build automation.
- [Codacy](https://app.codacy.com/welcome/organizations): rigorous code analysis.
- [Coveralls](https://coveralls.io/): for measuring and uploading [code coverage](https://coveralls.io/github/JamesYang007/FastAD).
- [Cpp Coveralls](https://github.com/eddyxu/cpp-coveralls): for measuring code coverage in Coveralls.
- [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page): matrix library that we wrapped to create AD expressions.
- [GCC](https://gcc.gnu.org/): compiler used to develop in linux environment.
- [Github Changelog Generator](https://github.com/github-changelog-generator/github-changelog-generator): generate [CHANGELOG](https://github.com/JamesYang007/FastAD/blob/master/CHANGELOG.md).
- [Google Benchmark](https://github.com/google/benchmark): benchmark against various methods.
- [GoogleTest](https://github.com/google/googletest): unit-test and integration-test.
- [Travis](https://travis-ci.org/): continuous integration for Linux and MacOS. See [.travis.yml](https://github.com/JamesYang007/FastAD/blob/master/.travis.yml) for more details.
- [Valgrind](http://valgrind.org/): check memory leak/error.

## License

- **[MIT license](http://opensource.org/licenses/mit-license.php)**
- Copyright 2019 ©JamesYang007.
