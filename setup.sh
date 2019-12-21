#!/bin/sh

# Setup google benchmark and googletest
if [ ! -d "libs/benchmark/googletest" ]; then
    mkdir build
    git clone https://github.com/google/googletest.git libs/benchmark/googletest
    cd libs/benchmark
    mkdir build && cd build
    cmake ../
    make -j6
    make test
else
    echo "googletest already exists in libs/benchmark. To setup again, first remove libs/benchmark/googletest."
fi
