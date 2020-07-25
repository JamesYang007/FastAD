#!/bin/bash

projectdir=$(dirname "BASH_SOURCE")
gbenchpath="libs/benchmark"
gtestpath="$gbenchpath/googletest"
eigen3path="libs/eigen-3.3.7"

# Update submodule if needed
git submodule update --init

# setup googletest
if [ ! -d "$gtestpath" ]; then
    git clone https://github.com/google/googletest.git $gtestpath
    cd $gtestpath &> /dev/null
    git checkout -q release-1.10.0
    cd ~- # change back to previous dir and no output to terminal
fi

# install Eigen3.3.7
if [ ! -d "$eigen3path" ]; then
    cd libs &> /dev/null
    wget https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.tar.gz
    tar xzf eigen-3.3.7.tar.gz
    cd eigen-3.3.7
    mkdir -p build && cd build
    cmake .. -DCMAKE_INSTALL_PREFIX="." # installs into build directory
    make install
fi
