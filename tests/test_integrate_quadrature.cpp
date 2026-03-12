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
    { return x; };

    Result r = integrate_trapezoid(0.0, 1.0, f, 10000, 5);
    assert(approx_equal(r.value, 0.5, r.error));
}

void test_integrate_simpson()
{
    // TODO

    auto f = [](double x)
    { return x; };

    Result r = integrate_simpson(0.0, 1.0, f, 10000, 5);

    assert(r.value == 0.0);
    assert(r.error == 0.0);
}

int main()
{
    test_integrate_trapezoid();
    test_integrate_simpson();
}
