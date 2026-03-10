#!/usr/bin/env bash

set -e

BUILD_DIR=build
TEST_BIN="$BUILD_DIR/test_integrate"

if [ ! -d "$BUILD_DIR" ]; then
    echo "Build directory not found. Building project first..."
    ./build.sh
fi

if [ ! -f "$TEST_BIN" ]; then
    echo "Test binary not found. Building project first..."
    ./build.sh
fi

echo "Running tests..."
ctest --test-dir $BUILD_DIR --output-on-failure