#include <cmath>

constexpr double kRelativeErrorTolerance = 1e-1;

bool approx_equal(double value, double expected, double error)
{
    return std::abs(value - expected) <= error * 2;
}

bool relative_equal(double value, double expected)
{
    if (std::abs(expected) < 1e-5) {
        return std::abs(value - expected) <= kRelativeErrorTolerance;
    } else {
        return std::abs(value - expected) <= kRelativeErrorTolerance * std::abs(expected);
    }
}