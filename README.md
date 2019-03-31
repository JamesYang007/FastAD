# FastAD

## Description

FastAD is a C++ implementation of automatic differentiation providing both forward and reverse mode.
Reverse mode is based on expression template design and template metaprogramming.
Forward mode is computed without any expression templates.
FastAD was developed in Visual Studio 2017 using C++14 standard.

## Installation

Change directory to installation directory and run the following command:

```
git clone https://JamesYang007@bitbucket.org/JamesYang007/autodiff.git
```

## Dependencies

- [Boost](https://www.boost.org/users/download/)

### Additional Dependencies for *Test* and *examples*

- [armadillo](http://arma.sourceforge.net/download.html)
- [googletest](https://github.com/google/googletest)
- Implementation of LAPACK/BLAS 
	- [Intel MKL](https://software.intel.com/en-us/mkl/choose-download) (Windows, Intel machine)

Visual Studio 2017 users may use solutions located in folder *vs_autodiff*.
If LAPACK/BLAS is required, open Property page Configuration Properties/Intel Performance Libraries.
Set "Use Intel MKL" to the desired choice (sequential is recommended if parallel is not possible).
To build/run tests or examples, user must follow one of two options:

#### Option 1:

For the respective project, open Properties/ C/C++ /Additional Include Directories.
Change all directories corresponding to **Boost**, **armadillo**, **googletest** to the absolute directories on local machine.

#### Option 2:

Copy **armadillo/**, **boost_X_XX_X/**, **googletest-master/** into **autodiff/**.
Note that **include/** folder should be contained in **armadillo/** and **boost_X_XX_X/**.
The folder **googletest-master/** should contain only one folder: **googletest-master/**.

## Tutorial

See *example/* folder.

## Further Implementation

- Delta Function

## Author
- James Yang
