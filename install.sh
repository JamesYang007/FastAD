#!/bin/bash

./clean-build.sh release
cd build/release
sudo ninja install -j6
