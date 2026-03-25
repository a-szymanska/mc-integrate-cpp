/*
The sampling algorithms for importance Monte Carlo methods.
*/

#pragma once

#include <random>
#include <vector>
#include <functional>

// ------------ Metropolis-Hastings (MCMC) -----------

template <typename T = double>
class McmcSampler
{
public:
    // Continuous distribution constructor
    McmcSampler(
        T lower,
        T upper,
        std::function<double(T)> pdf)
        : lower(lower),
          upper(upper),
          pdf(std::move(pdf)),
          dist_continuous(lower, upper),
          sample(&McmcSampler<T>::sample_continuous)
    {}

    // Discrete distribution constructor
    McmcSampler(
        std::vector<T> values,
        std::vector<double> probs)
        : values(std::move(values)),
          probs(std::move(probs)),
          dist_discrete(0, values.size() - 1),
          sample(&McmcSampler<T>::sample_discrete)
    {}

    T operator()()
    {
        return (this->*sample)();
    }

private:
    using sample_fn = T (McmcSampler<T>::*)();
    sample_fn sample;

    static std::mt19937 mt;
    static std::uniform_real_distribution<double> dist_prob;

    std::uniform_int_distribution<int> dist_discrete;
    std::uniform_real_distribution<double> dist_continuous;

    constexpr static int kNumIterationsInit = 1000;
    constexpr static int kNumIterationsStep = 1;

    // ----------- Continuous case -----------

    T lower = 0.0;
    T upper = 0.0;
    std::function<double(T)> pdf;
    T cur_state = 0.0;

    // ----------- Discrete case -----------

    std::vector<T> values;
    std::vector<double> probs;
    int cur_idx = 0;

    T sample_continuous();
    T sample_discrete();
};


template <typename T>
std::mt19937 McmcSampler<T>::mt(std::random_device{}());

template <typename T>
std::uniform_real_distribution<double> McmcSampler<T>::dist_prob(0.0, 1.0);

template <typename T>
T McmcSampler<T>::sample_continuous()
{
    int n_iterations = kNumIterationsStep;
    if (cur_state == 0.0) {  // First sample
        cur_state = dist_continuous(mt);
        n_iterations = kNumIterationsInit;
    }

    while (n_iterations--) {
        double next_state = dist_continuous(mt);
        double p_accept = std::min(1.0, pdf(next_state) / pdf(cur_state));
        if (dist_prob(mt) <= p_accept) {
            cur_state = next_state;
        }
    }

    return cur_state;
}

template <typename T>
T McmcSampler<T>::sample_discrete()
{
    int n_iterations = kNumIterationsStep;
    if (cur_idx == 0.0) {  // First sample
        cur_idx = dist_discrete(mt);
        n_iterations = kNumIterationsInit;
    }

    while (n_iterations--) {
        int next_idx = dist_discrete(mt);
        double p_accept = std::min(1.0, probs[next_idx] / probs[cur_idx]);
        if (dist_prob(mt) <= p_accept) {
            cur_idx = next_idx;
        }
    }

    return values[cur_idx];
}
