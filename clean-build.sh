#!/bin/bash

# directory where current shell script resides
PROJECTDIR=$(dirname "$BASH_SOURCE")

cd "$PROJECTDIR"

mode=$1 # debug/release mode
shift   # shift command-line arguments
run=$1  # run ctest
shift   # other cmake arguments

mkdir -p build && cd build

# if debug directory does not exist, create it
mkdir -p debug
# if release directory does not exist, create it
mkdir -p release

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

rm -rf ./*

# if $run is set to "run" or not set to anything
if [ "$run" = "run" ] || [ "$run" = "" ]; then
    cmake -GNinja ../../ "$@"   # append other cmake command-line arguments
else
    cmake -GNinja ../../ "$run $@"   # append $run (assumed to be first cmake command-line argument if not run)
                             # and other cmake command-line arguments
fi

ninja -j12           # make with 12 threads

if [ "$run" = "run" ]; then
    ctest -j12
fi
