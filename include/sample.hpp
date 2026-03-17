/*
The sampling algorithms for importance Monte Carlo methods.
*/

#pragma once

#include <vector>
#include <functional>

// ------------ Metropolis-Hastings (MCMC) -----------

/**
 * Draws a sample from a continuous distribution defined by
 * a probability density function f over the interval [lower, upper].
 */
double sample_mcmc(
    double lower,
    double upper,
    const std::function<double(double)> &f,
    int n_iterations = 1000);

/**
 * Draws a sample from a discrete distribution defined by
 * a set of probabilities corresponding to discrete values in the domain.
 * 
 * The input distribution is represented as vector of values and a vector
 * of corresponding probabilities.
 */
double sample_mcmc(
    const std::vector<double> &values,
    const std::vector<double> &probs,
    int n_iterations = 1000);