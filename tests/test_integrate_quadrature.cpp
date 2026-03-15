#include "../include/integrate.hpp"
#include "./utils.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>
#include <string>

void test_integrate_trapezoid()
{
    auto f = [](double x)
    { return x * x; };

    Result r = integrate_trapezoid(0.0, 1.0, f, 1000, 5);
    assert(approx_equal(r.value, 1/3.0, r.error));
}

void test_integrate_simpson()
{
    auto f = [](double x)
    { return x * x; };

    Result r = integrate_simpson(0.0, 1.0, f, 1000, 5);
    assert(approx_equal(r.value, 1/3.0, r.error));
}

int main()
{
    test_integrate_trapezoid();
    test_integrate_simpson();
}
