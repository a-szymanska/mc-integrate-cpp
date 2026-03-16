/*
Omp parallelized versions of algorithms from integrate_mc, mmplementation of Monte Carlo methods for numerical integration,
with Vegas optimization and Welford's algorithm for error estimation. 
*/

#pragma once

#include <functional>

struct Result {
    double value;
    double error;
};

/*
The total number of sampled points is equal to n_points * n_iterations.
*/

/*
The returned error estimates the standard deviation of the integral estimate
based on the sample variance of the function in each bin.
*/


// ----- Monte Carlo integration parallelised with OMP (1-dimensional) -----

Result integrate_MC_omp(
    double lower,
    double upper,
    const std::function<double(double)> &f,
    int n_points,
    int n_boxes,
    int n_iterations);

// ----- Monte Carlo integration parallelised with OMP (N-dimensional) -----

Result integrate_MC_ndim_omp(
    const std::vector<double> &lower,
    const std::vector<double> &upper,
    const std::function<double(const std::vector<double> &)> &f,
    int n_points);

// ------ Monte Carlo with custom distribution, parallelised with OMP ------

Result integrate_MC_dist_omp(
    double lower,
    double upper,
    const std::function<double(double)> &f,
    const std::function<double(double)> &p,
    int n_points,
    int n_boxes,
    int n_iterations);
