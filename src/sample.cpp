/*
Implementation of the sampling algorithms for importance Monte Carlo methods.
*/

#include "../include/sample.hpp"

#include <random>
#include <cmath>

double sample_mcmc(
    double lower,
    double upper,
    const std::function<double(double)> &f,
    int n_iterations)
{
    static std::mt19937 mt(std::random_device{}());
    std::uniform_real_distribution<double> dist(lower, upper);
    std::uniform_real_distribution<double> dist_bool(0.0, 1.0);

    double q = dist(mt);
    while (n_iterations--) {
        double q_next = dist(mt);
        double p_accept = std::min(1.0, f(q_next) / f(q));
        if (dist_bool(mt) <= p_accept) {
            q = q_next;
        }
    }

    return q;
}

double sample_mcmc(
    const std::vector<double> &values,
    const std::vector<double> &probs,
    int n_iterations)
{
    static std::mt19937 mt(std::random_device{}());
    std::uniform_int_distribution<int> dist(0, values.size() - 1);
    std::uniform_real_distribution<double> dist_bool(0.0, 1.0);

    int q = dist(mt);
    while (n_iterations--) {
        int q_next = dist(mt);
        double p_accept = std::min(1.0, probs[q_next] / probs[q]);
        if (dist_bool(mt) <= p_accept) {
            q = q_next;
        }
    }

    return values[q];
}