#!/bin/sh

# If setup.sh was called before
if [ -d "libs/benchmark/googletest" ]; then
    rm -rf libs/benchmark
fi

# Update submodule if needed
git submodule update --remote
# Setup google benchmark and googletest
git clone https://github.com/google/googletest.git libs/benchmark/googletest
cd libs/benchmark
mkdir -p build && cd build
cmake ../ -GNinja
cmake --build . -- -j12
