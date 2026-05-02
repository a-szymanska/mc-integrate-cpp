#!/usr/bin/env bash

set -e

BUILD_DIR=build
USE_GNUPLOT=""
USE_OMP=OFF

# Check for flags
for arg in "$@"; do
    if [[ "$arg" == "--gnuplot" ]]; then
        USE_GNUPLOT=ON
    fi
    if [[ "$arg" == "--omp" ]]; then
        USE_OMP=ON
    fi
done

# If no --gnuplot flag, ask the user
if [[ -z "$USE_GNUPLOT" ]]; then
    read -p "Download and enable gnuplot? (y/n) [y]: " answer
    answer=${answer:-y}

    if [[ "$answer" =~ ^[Yy]$ ]]; then
        USE_GNUPLOT=ON
    else
        USE_GNUPLOT=OFF
    fi
fi

echo "Configuring project..."
cmake -B $BUILD_DIR -DUSE_GNUPLOT=$USE_GNUPLOT -DUSE_OMP=$USE_OMP -Wno-dev

echo "Building..."
cmake --build $BUILD_DIR

echo "Running tests..."
ctest --test-dir $BUILD_DIR --output-on-failure