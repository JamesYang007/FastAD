#!/bin/bash

./clean-build.sh release
cd build
cd release
make install -j6
