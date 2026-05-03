#include <cmath>

constexpr double kRelativeErrorTolerance = 1e-1;

bool approx_equal(double value, double expected, double error)
{
    return std::fabs(value - expected) <= error * 2;
}

bool relative_equal(double value, double expected)
{
    if (std::abs(expected) < 1e-5) {
        return std::fabs(value - expected) <= kRelativeErrorTolerance;
    } else {
        return std::fabs(value - expected) <= kRelativeErrorTolerance * std::abs(expected);
    }
}
