# FastAD

## Description

FastAD is a C++ implementation of automatic differentiation providing both forward and reverse mode.
Reverse mode is based on expression template design and template metaprogramming.
Forward mode is computed without any expression templates.
FastAD supports build in Visual Studio 2017 and Linux/MacOS.

## Installation

Change directory to installation directory and run the following command:

```
git clone --recurse-submodules https://github.com/JamesYang007/FastAD.git 
```

## Dependencies

- [Boost](https://www.boost.org/users/download/)

### Additional Dependencies 

Although FastAD is generic, certain features assume an interface for a matrix library.
We modeled after `armadillo` in terms of syntax.
It is recommended that the user installs `armadillo` to compute Jacobian and Hessian.

- [armadillo](http://arma.sourceforge.net/download.html)
- Implementation of LAPACK/BLAS 
	- [Intel MKL](https://software.intel.com/en-us/mkl/choose-download) (Windows, Intel machine)

Visual Studio 2017 users may use solutions located in `vs_autodiff` directory.
If `LAPACK`/`BLAS` is required, open `Property page` and go to `Configuration Properties/Intel Performance Libraries`.
Set `Use Intel MKL` to the desired choice (sequential is recommended if parallel is not possible).

## Build and Run

### Windows

For the respective project, open `Properties/ C/C++ /Additional Include Directories`.
Change all directories corresponding to `Boost`, `armadillo`, `googletest` to the absolute directories on local machine.

### Linux/MacOS

It is recommended to perform a clean build and specify the mode (`debug` or `release`).
Simply run the following to build only:

```
./clean-build.sh <mode>
```

To execute the tests run the following:

```
./clean-build.sh <mode> run
```

Alternatively,

```
./clean-build.sh <mode>
cd build/<mode>
ctest -j12
```

## Tutorial

See `example` directory.

## Further Implementation

- Delta Function

## Author

- James Yang
