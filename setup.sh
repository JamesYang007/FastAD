#!/bin/bash

projectdir=$(dirname "BASH_SOURCE")
gbenchpath="libs/benchmark"
gtestpath="$gbenchpath/googletest"
eigen3path="libs/eigen-3.3.7"

is_dev=$1

# install Eigen3.3.7
if [ ! -d "$eigen3path" ]; then
    cd libs &> /dev/null
    wget https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.tar.gz
    tar xzf eigen-3.3.7.tar.gz
    cd eigen-3.3.7
    mkdir -p build && cd build
    cmake .. -DCMAKE_INSTALL_PREFIX="." # installs into build directory
    make install
    cd ../../../ &> /dev/null
fi

# setup only for devs
if [ "$is_dev" == "dev" ]; then

    echo "Setting up extra dev tools..."

    # setup google benchmark
    if [ ! -d "$gbenchpath" ]; then
        git clone https://github.com/google/benchmark.git $gbenchpath
        cd $gbenchpath &>/dev/null
        git checkout -q v1.5.0
        cd ~- # change back to previous dir and no output to terminal
    fi

    # setup googletest
    if [ ! -d "$gtestpath" ]; then
        git clone https://github.com/google/googletest.git $gtestpath
        cd $gtestpath &> /dev/null
        git checkout -q release-1.10.0
        cd ~- # change back to previous dir and no output to terminal
    fi

fi # end dev-only
