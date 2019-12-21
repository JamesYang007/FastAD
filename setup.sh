#!/bin/sh

# Setup google benchmark and googletest
git clone https://github.com/google/googletest.git libs/benchmark/googletest
cd libs/benchmark
mkdir build && cd build
cmake ../
make -j6
make test
