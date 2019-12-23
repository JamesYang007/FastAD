# How to contribute

Here is a guideline for reporting any bugs, issues, changes, or additions.

## Prerequisites

First check the [issue page](https://github.com/JamesYang007/FastAD/issues) to see if your issue already exists.
If not, [create a new issue](https://github.com/nlohmann/json/issues/new/choose) 
and describe the issue as thoroughly as possible. 
User must have a [GitHub account](https://github.com/signup/free) to create a new issue.

## Description

The following is a rough outline of how to thoroughly describe an issue:

### Bug

Please provide a minimal example to recreate the bug along with the error message.
Additionally, state what you **expected** to happen instead of the error.

### Change or Addition

Explain what will be changed, motivation for doing so, and changes in use-cases.

### Compiler Error

Please specify the compiler, version, and operating system along with an attachment of the error message.

## Files to change

Please create a separate branch before proceeding.
Branch naming scheme is as follows: `<first-name>.<last-name>/<feature>`.
As an example, `james.yang/fix_exgen_feval_function`.

To make changes, you need to edit the following files:

1. [`include/fastad_bits`](https://github.com/JamesYang007/FastAD/tree/master/include/fastad_bits)

    The entire library is in this directory, i.e. all headers are stored here.

2. [`test`](https://github.com/JamesYang007/FastAD/tree/master/test)

    These files contain all unit and integration tests.

If you add or change a feature, please also add a unit test to this file. The unit tests can be compiled and executed with

```sh
$ ./clean-build debug
$ cd build/debug
$ ctest
```

The test cases are also executed on [Travis](https://travis-ci.org/nlohmann/json) once you open a pull request.
