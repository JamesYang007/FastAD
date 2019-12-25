#!/bin/sh

# If setup.sh was called before
if [ -d "libs/benchmark/googletest" ]; then
    rm -rf libs/benchmark
    git submodule update --remote
fi

# Setup google benchmark and googletest
git clone https://github.com/google/googletest.git libs/benchmark/googletest
cd libs/benchmark
mkdir -p build && cd build
cmake -GNinja ../
ninja -j12
