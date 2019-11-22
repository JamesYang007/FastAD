#!/bin/bash

# directory where current shell script resides
PROJECTDIR=$(dirname "$BASH_SOURCE")

cd $PROJECTDIR

mode=$1 # debug/release mode
shift   # shift command-line arguments
run=$1  # run ctest
shift

# other cmake arguments

# if build directory does not exist, create it
if [ ! -d "build" ]; then
    mkdir build
fi

# change directory to build
cd build

# if debug directory does not exist, create it
if [ ! -d "debug" ]; then
    mkdir debug
fi
cd debug && rm -rf * && cd ..

# if release directory does not exist, create it
if [ ! -d "release" ]; then
    mkdir release
fi
cd release && rm -rf * && cd ..

# if debug mode
if [ "$mode" = "debug" ]; then
    cd debug
# if release mode
elif [ "$mode" = "release" ]; then
    cd release
else
    echo "Usage: ./clean-build.sh <debug/release> [run] ..." 1>&2
    exit 1
fi

cmake ../../ "$@"   # append other cmake command-line arguments
make -j12           # make with 12 threads

if [ "$run" = "run" ]; then
    ctest -j12
fi
