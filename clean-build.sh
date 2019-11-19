#!/bin/bash

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
