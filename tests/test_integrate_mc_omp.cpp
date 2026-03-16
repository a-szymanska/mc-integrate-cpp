#include "../include/integrate_mc_omp.hpp"
#include "./utils.hpp"

#include <cassert>
#include <vector>

void test_integrate_MC_omp()
{
    auto f = [](double x) { return x; };

    Result r = integrate_MC_omp(0.0, 1.0, f, 10000, 10, 10);
    assert(approx_equal(r.value, 0.5, r.error));
}

void test_integrate_MC_ndim_omp()
{
    // TODO

    std::vector<double> lower = {0.0, 0.0};
    std::vector<double> upper = {1.0, 1.0};
    auto f = [](const std::vector<double> &x)
        { return x[0] + x[1]; };

    Result r = integrate_MC_ndim_omp(lower, upper, f, 100000);

    assert(r.value == 0.0);
    assert(r.error == 0.0);
}

void test_integrate_MC_dist_omp()
{
    // TODO

    auto f = [](double x) { return x; };
    auto p = [](double x) { return 1.0; };

    Result r = integrate_MC_dist_omp(0.0, 1.0, f, p, 1000, 10, 5);

    assert(r.value == 0.0);
    assert(r.error == 0.0);
}


int main()
{
    test_integrate_MC_omp();
    test_integrate_MC_ndim_omp();
    test_integrate_MC_dist_omp();
}
