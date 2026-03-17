#!/usr/bin/env bash
set -e

OUT="mc_integrate_cpp.hpp"

echo "Generating $OUT..."

{
echo "/*"
echo "  mc_integrate_cpp.hpp - amalgamated version"
echo "  https://github.com/a-szymanska/mc-integrate-cpp"
echo "  Distributed under the MIT License"
echo "  Generated on $(date)"
echo "*/"
echo
echo "#pragma once"
echo
} > "$OUT"

# System includes without duplicates
grep -h '^#include <.*>$' include/*.hpp src/*.cpp | sort -u >> "$OUT"
echo >> "$OUT"

# Include common.hpp
if [ -f include/common.hpp ]; then
    echo "// ===== include/common.hpp =====" >> "$OUT"
    grep -v -E '^#include |^#pragma once' include/common.hpp >> "$OUT"
    echo >> "$OUT"
fi

# Include all other headers
for f in $(ls include/*.hpp | grep -v -E 'common.hpp|_omp'); do
    echo "// ===== $f =====" >> "$OUT"
    grep -v -E '^#include |^#pragma once' "$f" >> "$OUT"
    echo >> "$OUT"
done

# Include source code
for f in $(ls src/*.cpp | grep -v '_omp'); do
    echo "// ===== $f =====" >> "$OUT"
    grep -v -E '^#include |^#pragma once' "$f" >> "$OUT"
    echo >> "$OUT"
done

echo "The header successfully generated."