# Monte Carlo integration for C++

Single-header C++ library for Monte Carlo (and not only) integration with support for multidimensional integrals and optional multithreading for faster computation.

## Integration methods

### Monte Carlo methods
- `integrate_MC` - integration in 1D with Vegas optimisation
- `integrate_MC_ndim` - integration for multidimensional integrals
- `integrate_MC_highdim` - integration for multidimensional integrals, optimised for large number of dimensions 
- `integrate_MC_dist` - integration with importance sampling

See [`include/integrate_mc.hpp`](include/integrate_mc.hpp).

### Quadrature methods
- `integrate_trapezoid` - trapezoidal method with adaptive box partition
- `integrate_simpson` - Simpson's 1/3 trapezoidal method with adaptive box partition

See [`include/integrate_quadrature.hpp`](include/integrate_quadrature.hpp).

## Get started

### Build
To configure the project and compile tests run:
```bash
./build.sh
```

### Usage

To bundle the library into a single header:
```bash
./amalgamate.sh
```
To use the library in your own project download the generated all-in-one header in `mc_integrate_cpp.hpp`.

### Examples

See the `tests/` directory for example usage of all the integration functions.
