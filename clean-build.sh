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

# if release directory does not exist, create it
if [ ! -d "release" ]; then
    mkdir release
fi

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

rm -rf *

# if $run is set to "run" or not set to anything
if [ "$run" = "run" ] || [ "$run" = "" ]; then
    cmake ../../ "$@"   # append other cmake command-line arguments
else
    cmake ../../ "$run $@"   # append $run (assumed to be first cmake command-line argument if not run)
                             # and other cmake command-line arguments
fi

make -j12           # make with 12 threads

if [ "$run" = "run" ]; then
    ctest -j12
fi
