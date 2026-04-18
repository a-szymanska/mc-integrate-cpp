/*
Implementation of Monte Carlo methods for numerical integration,
with Vegas optimization and Welford's algorithm for error estimation.
*/

#pragma once

#include "common.hpp"
#include "estimators.hpp"
#include <vector>
#include <functional>

/*
The total number of sampled points is equal to n_points * n_iterations.
*/

/*
The returned error estimates the standard deviation of the integral estimate
based on the sample variance of the function in each bin.
*/

// ----- Monte Carlo integration (1-dimensional) -----

Result integrate_MC(
    double lower,
    double upper,
    const std::function<double(double)> &f,
    int n_points,
    int n_bins,
    int n_iterations);

// ----- Monte Carlo integration (N-dimensional) -----

template <typename Estimator = EstimatorSimple>
Result integrate_MC_ndim(
    const std::vector<double> &lower,
    const std::vector<double> &upper,
    const std::function<double(const std::vector<double> &)> &f,
    int n_bins,
    int burn_in_size,
    int n_points);


// ----- Monte Carlo integration (N-dimensional) optimised for high dimension count -----

template <typename Estimator = EstimatorSimple>
Result integrate_MC_highdim(
    const std::vector<double> &lower,
    const std::vector<double> &upper,
    const std::function<double(const std::vector<double> &)> &f,
    int n_bins,
    int burn_in_size,
    int n_points);

// ------ Monte Carlo with custom distribution ------

template <typename Estimator = EstimatorSimple>
Result integrate_MC_dist(
    double lower,
    double upper,
    const std::function<double(double)> &f,
    const std::function<double(double)> &pdf,
    int n_points);
