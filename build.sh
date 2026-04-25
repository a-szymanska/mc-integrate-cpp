#!/usr/bin/env bash

set -e

BUILD_DIR=build

read -p "Download and enable gnuplot? (y/n) [y]: " answer
answer=${answer:-y}

if [[ "$answer" =~ ^[Yy]$ ]]; then
    USE_GNUPLOT=ON
else
    USE_GNUPLOT=OFF
fi

echo "Configuring project..."
cmake -B $BUILD_DIR -DUSE_GNUPLOT=$USE_GNUPLOT

echo "Building..."
cmake --build $BUILD_DIR

echo "Running tests..."
ctest --test-dir $BUILD_DIR --output-on-failure