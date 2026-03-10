#!/usr/bin/env bash

set -e

BUILD_DIR=build

echo "Configuring project..."
cmake -B $BUILD_DIR

echo "Building..."
cmake --build $BUILD_DIR

echo "Running tests..."
ctest --test-dir $BUILD_DIR --output-on-failure