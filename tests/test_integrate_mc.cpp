#include "../include/integrate_mc.hpp"
#include "./utils.hpp"
#include "sample_area.hpp"

#include <cassert>
#include <cmath>
#include <vector>
#include <iostream>
void test_integrate_MC()
{
    auto f = [](double x) { return x; };

    Result r = integrate_MC(0.0, 1.0, f, 100000, 100, 20);
    std::cout<<r.value<<" "<<r.error<<std::endl;
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

    Result r = integrate_MC_ndim<EstimatorSimple, AreaSampler>(lower, upper, f, 100000, 4, 1000);

    std::cout<<r.value<<" "<<r.error<<std::endl;
    assert(approx_equal(r.value, 32, r.error));
}

void test_integrate_MC_highdim()
{
    std::vector<double> lower(3, 0);
    std::vector<double> upper(3, M_PI);

    auto f = [](const std::vector<double> &X) { 
        double res = 1;
        for (double val: X) {
            res *= std::sin(val);
        }
        return res;
    };
    Result r = integrate_MC_ndim<EstimatorSimple, BinSampler>(lower, upper, f, 100000, 10, 1000);

    std::cout<<r.value<<" "<<pow(2.0,3.0)<<" "<<r.error<<std::endl;
    assert(approx_equal(r.value, pow(2.0,3.0), r.error));
}

void test_integrate_MC_dist()
{
    auto f = [](double x) { return std::sin(x); };
    auto p = [](double x) { return 8 * x / (M_PI * M_PI); }; // Linear PDF normalized over [0, pi/2]


    Result r = integrate_MC_dist<EstimatorSimple>(0.0, M_PI/2, f, p, 100000);

    std::cout<<r.value<<" "<<r.error<<std::endl;
    assert(approx_equal(r.value, 1.0, r.error));
}

int main()
{
    test_integrate_MC();
    test_integrate_MC_ndim();
    test_integrate_MC_highdim();
    test_integrate_MC_dist();
}
