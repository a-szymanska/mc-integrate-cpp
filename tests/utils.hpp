#include <cmath>

bool approx_equal(double value, double expected, double error)
{
    return std::abs(value - expected) <= error * 1.5;
}
