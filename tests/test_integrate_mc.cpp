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

    auto f = [](const std::vector<double> &x)
        { 
          double res = 1;
          for(double cur: x){ res*=std::sin(cur);}
          return res;
        };

    Result r = integrate_MC_ndim(lower, upper, f, 10, 100000);

    assert(approx_equal(r.value, 32, r.error));
}

void test_integrate_MC_dist()
{
    // TODO

    auto f = [](double x) { return x; };
    auto p = [](double x) { return 1.0; };

    Result r = integrate_MC_dist(0.0, 1.0, f, p, 10000, 10, 10);

    assert(r.value == 0.0);
    assert(r.error == 0.0);
}


int main()
{
    test_integrate_MC();
    test_integrate_MC_ndim();
    test_integrate_MC_dist();
}
