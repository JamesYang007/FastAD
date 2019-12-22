#!/bin/sh

# Setup google benchmark and googletest
if [ ! -d "libs/benchmark/googletest" ]; then
    git clone https://github.com/google/googletest.git libs/benchmark/googletest
    cd libs/benchmark
    mkdir build && cd build
    cmake ../ -DCMAKE_BUILD_TYPE=Release -DBENCHMARK_ENABLE_LTO=true
    make -j6
else
    echo "googletest already exists in libs/benchmark. To setup again, first remove libs/benchmark/googletest."
fi
