#include <cmath>

constexpr double kToleranceError = 1e-3;

bool approx_equal(double value, double expected, double error)
{
    return error <= kToleranceError && std::abs(value - expected) <= error * 1.5;
}
