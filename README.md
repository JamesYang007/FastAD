# FastAD [![Build Status](https://travis-ci.org/JamesYang007/FastAD.svg?branch=master)](https://travis-ci.org/JamesYang007/FastAD) [![License](http://img.shields.io/:license-mit-blue.svg?style=flat-square)](http://badges.mit-license.org)

> FastAD is a light-weight, header-only C++ library of automatic differentiation providing both forward and reverse mode.

---

## Table of contents
* [Example](#example)
* [Installation](#installation)
* [Features](#features)
* [Tests](#tests)
* [Team](#team)
* [FAQ](#faq)
* [Support](#support)
* [License](#license)

---

## Example

---

## Installation
Change directory to installation directory and run the following command:

### Clone
```shell
$ git clone --recurse-submodules https://github.com/JamesYang007/FastAD.git 
```

### Setup
* Install a compiler with full support for C++17 standard.
* Recommended to install [armadillo](http://arma.sourceforge.net/download.html) and its dependencies.
* (Linux/MacOS) Install cmake >= 3.9

---

## Features
* Forward-mode automatic differentiation
* Reverse-mode automatic differentiation
* Easy way to compute and store jacobian
* Easy way to compute and store hessian
* Supports differentiation of polynomials and most elementary functions in standard library

---

## Tests

### Linux/MacOS
> to build
```
mkdir build && cd build
cmake ../
make
```

> to run tests
```
ctest
```

---

## Team

| **JamesYang007** |
| :---: |
| [![JamesYang007](https://avatars3.githubusercontent.com/u/5008832?s=100&v=4)](https://github.com/JamesYang007) |
| <a href="http://github.com/JamesYang007" target="_blank">`github.com/JamesYang007`</a> |

---

## FAQ

* **How do I build my project with FastAD as a dependency**?

---

## Support
Feel free to contact me via:
* Email: jamesyang916 at gmail.com

---

## License

[![License](http://img.shields.io/:license-mit-blue.svg?style=flat-square)](http://badges.mit-license.org)

- **[MIT license](http://opensource.org/licenses/mit-license.php)**
- Copyright 2015 ©JamesYang007.
