#include "../include/integrate_mc.hpp"
#include "./utils.hpp"

#include <cassert>
#include <cmath>
#include <vector>

void test_integrate_MC()
{
    auto f = [](double x) { return x; };

    Result r = integrate_MC(0.0, 1.0, f, 1000, 10, 5);
    assert(approx_equal(r.value, 0.5, r.error));
}

void test_integrate_MC_ndim()
{
    std::vector<double> lower(5, 0);
    std::vector<double> upper(5, M_PI);

    auto f = [](const std::vector<double> &X) { 
        double res = 1;
        for (double val: X) {
            res *= std::sin(val);
        }
        return res;
    };

    Result r = integrate_MC_ndim(lower, upper, f, 10, 20, 100000);
    assert(approx_equal(r.value, 32, r.error));
}

void test_integrate_MC_dist()
{
    auto f = [](double x) { return std::sin(x); };
    auto p = [](double x) { return 8 * x / (M_PI * M_PI); }; // Linear PDF normalized over [0, pi/2]

    Result r = integrate_MC_dist(0.0, M_PI/2, f, p, 10000);
    assert(approx_equal(r.value, 1.0, r.error));
}

int main()
{
    test_integrate_MC();
    test_integrate_MC_ndim();
    test_integrate_MC_dist();
}
