/*
The sampling algorithms for importance Monte Carlo methods.
*/

#pragma once

#include <random>
#include <vector>
#include <functional>

// ------------ Metropolis-Hastings (MCMC) -----------

class McmcSampler
{
public:
    // Continuous distribution constructor
    McmcSampler(
        double lower,
        double upper,
        std::function<double(double)> pdf)
        : lower(lower),
          upper(upper),
          pdf(std::move(pdf)),
          sample(&McmcSampler::sample_continuous)
    {}

    // Discrete distribution constructor
    McmcSampler(
        std::vector<double> values,
        std::vector<double> probs)
        : values(std::move(values)),
          probs(std::move(probs)),
          sample(&McmcSampler::sample_discrete)
    {}

    double operator()()
    {
        return (this->*sample)();
    }

private:
    using sample_fn = double (McmcSampler::*)();
    sample_fn sample;

    static std::mt19937 mt;
    static std::uniform_real_distribution<double> dist_bool;

    constexpr static int kIterationsInit = 1000;
    constexpr static int kIterationsStep = 10;

    // ----------- Continuous case -----------

    double lower = 0.0;
    double upper = 0.0;
    std::function<double(double)> pdf;
    double cur_state = 0.0;

    // ----------- Discrete case -----------

    std::vector<double> values;
    std::vector<double> probs;
    int cur_idx = 0;

    double sample_continuous();
    double sample_discrete();
};
