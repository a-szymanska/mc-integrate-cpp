# mc-integrate-cpp

Single-header C++ library for Monte Carlo integration with support for multidimensional integrals and optional multithreading for faster computation.

## Get started

### Build
To configure the project and compile tests run:
```bash
./build.sh
```

### Usage

To use the library in your own project download the header in `include/integrate.hpp`.
```cpp
#include "integrate.hpp"
```

### Examples

See the `tests/` directory for example usage of all integration functions:

- `integrate_MC`
- `integrate_MC_ndim`
- `integrate_MC_dist`
- `integrate_trapezoid`