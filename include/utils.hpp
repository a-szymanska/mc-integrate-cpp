/*
    Gelman-Rubin diagnostic for estimating the time needed to achieve
    a stationary distribution in MCMC sampling.

    Note: This is a heuristic and should be evaluated several times
    to obtain a more reliable estimate of convergence.
*/

#pragma once

#include <vector>
#include <functional>

#include "sample_mcmc.hpp"

constexpr double kGelmanRubinConverge = 1.05; // Threshold for convergence

template <typename T>
int estimate_burn_in_helper(McmcSampler<T>&, McmcSampler<T>&);

// ----------------- Continuous case -----------------
int estimate_burn_in(double lower, double upper, std::function<double(double)> pdf)
{
    McmcSampler<> sampler1(lower, upper, pdf, lower, 1); // Start at lower and upper
    McmcSampler<> sampler2(lower, upper, pdf, upper, 1);

    auto t = estimate_burn_in_helper(sampler1, sampler2);

    return t;
}

// ------------------ Discrete case ------------------
template <typename T>
int estimate_burn_in(std::vector<T> &values, std::vector<double> &probs)
{
    McmcSampler<T> sampler1(values, probs, 0, 1); // Start at first and last index
    McmcSampler<T> sampler2(values, probs, static_cast<int>(values.size() - 1), 1);

    auto t = estimate_burn_in_helper(sampler1, sampler2);

    return t;
}

// ----------------- Helper function -----------------
template <typename T>
int estimate_burn_in_helper(McmcSampler<T> &sampler1, McmcSampler<T> &sampler2)
{
    const int n_iterations_init = 100;

    double sum1 = 0, sum2 = 0;
    double sum_sq1 = 0, sum_sq2 = 0;
    int t = 1;
    double r = 10; // Start wirh r > kGelmanRubinConverge
    do {
        auto x1 = sampler1();
        auto x2 = sampler2();
        sum1 += x1;
        sum2 += x2;
        sum_sq1 += x1 * x1;
        sum_sq2 += x2 * x2;

        t++;
        double mean1 = sum1 / t;
        double mean2 = sum2 / t;
        double total_mean = (mean1 + mean2) / 2.0;

        // Between-chain variance
        double b = t * ((mean1 - total_mean) * (mean1 - total_mean) + (mean2 - total_mean) * (mean2 - total_mean));

        // Within-chain variance
        double w = ((sum_sq1 - sum1 * sum1 / t) + (sum_sq2 - sum2 * sum2 / t)) / (2.0 * (t - 1));

        r = ((t - 1) * w + b) / (t * w);
    } while (t < n_iterations_init || r > kGelmanRubinConverge);

    return t;
}