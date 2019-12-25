#!/bin/bash

./clean-build.sh release
cd build/release
ninja install -j6
