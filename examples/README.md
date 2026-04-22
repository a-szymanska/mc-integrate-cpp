# Examples

This directory contains example programmes demonstrating how to use the library in simulations and some other practical use-cases.

- Ising Model (see `ising.cpp`) - a statistical discrete system with local interactions

Modify sample sizes and simulation parameters to explore the trade-off between accuracy and performance.

## Build

From the root of the repository run:
```sh
./build.sh
```
This will compile all examples and place the resulting binaries in `examples/build/`.

## Visualization with gnuplot

The examples can optionally produce the plots using gnuplot. The build scirpt automatically downloads a minimal gnuplot source in a single-header file (saved in `build/external/gnuplot-iostream.h`).