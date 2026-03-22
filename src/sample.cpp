/*
Implementation of the sampling algorithms for importance Monte Carlo methods.
*/

#include "../include/sample.hpp"

// ------------ Metropolis-Hastings (MCMC) -----------

std::mt19937 McmcSampler::mt(std::random_device{}());
std::uniform_real_distribution<double> McmcSampler::dist_bool(0.0, 1.0);

double McmcSampler::sample_continuous()
{
    std::uniform_real_distribution<double> dist(lower, upper);

    int n_iterations = kIterationsStep;
    if (cur_state == 0.0) {  // First sample
        cur_state = dist(mt);
        n_iterations = kIterationsInit;
    }

    while (n_iterations--) {
        double next_state = dist(mt);
        double p_accept = std::min(1.0, pdf(next_state) / pdf(cur_state));
        if (dist_bool(mt) <= p_accept) {
            cur_state = next_state;
        }
    }

    return cur_state;
}

double McmcSampler::sample_discrete()
{
    std::uniform_int_distribution<int> dist(0, values.size() - 1);
    
    int n_iterations = 1;
    if (cur_idx == 0.0) {  // First sample
        cur_idx = dist(mt);
        n_iterations = 100;
    }

    while (n_iterations--) {
        int next_idx = dist(mt);
        double p_accept = std::min(1.0, probs[next_idx] / probs[cur_idx]);
        if (dist_bool(mt) <= p_accept) {
            cur_idx = next_idx;
        }
    }
    
    return values[cur_idx];
}