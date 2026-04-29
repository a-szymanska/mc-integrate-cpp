/*
The sampling algorithms for importance Monte Carlo methods.
*/

#pragma once

#include <random>
#include <vector>
#include <functional>
#include <optional>
#include <type_traits>

constexpr static int kNumIterationsInit = 1000;

// ------------------ Metropolis-Hastings (MCMC) ------------------

template <typename T = double>
class McmcSampler
{
public:
    // Continuous distribution constructor
    // implemented only for T = double
    template <typename U = T, typename = std::enable_if_t<std::is_same<U, double>::value>>
    McmcSampler(double lower, double upper, std::function<double(double)> pdf);

    template <typename U = T, typename = std::enable_if_t<std::is_same<U, double>::value>>
    McmcSampler(double lower, double upper, std::function<double(double)> pdf, double init_value, int n_iterations_init = kNumIterationsInit);

    // Discrete distribution constructor
    McmcSampler(std::vector<T> &values, std::vector<double> &probs);
    McmcSampler(std::vector<T> &values, std::vector<double> &probs, int init_idx, int n_iterations_init = kNumIterationsInit);

    T operator()()
    {
        return (this->*sample)();
    }

    
private:
    static std::mt19937 mt;
    std::uniform_real_distribution<double> dist_prob;

    using sample_fn = T (McmcSampler<T>::*)();
    sample_fn sample;

    // ----------- Continuous case -----------

    double lower;
    double upper;
    std::function<double(double)> pdf;
    double cur_value;

    std::uniform_real_distribution<double> dist_continuous;
    double sample_continuous();

    // ----------- Discrete case -----------

    std::optional<std::reference_wrapper<std::vector<T>>> values;
    std::optional<std::reference_wrapper<std::vector<double>>> probs;
    int cur_idx;

    std::uniform_int_distribution<int> dist_discrete;
    T sample_discrete();
};


template <typename T>
std::mt19937 McmcSampler<T>::mt{std::random_device{}()};

template <typename T>
template <typename U, typename>
McmcSampler<T>::McmcSampler(double lower, double upper, std::function<double(double)> pdf)
    : McmcSampler(lower, upper, pdf, std::uniform_real_distribution<double>(lower, upper)(mt))
{}

template <typename T>
template <typename U, typename>
McmcSampler<T>::McmcSampler(double lower, double upper, std::function<double(double)> pdf, double init_value, int n_iterations_init)
    : lower(lower),
      upper(upper),
      pdf(pdf),
      dist_continuous(lower, upper),
      dist_prob(0.0, 1.0),
      sample(&McmcSampler<T>::sample_continuous)
{
    cur_value = init_value;

    for (int i = 0; i < n_iterations_init; i++) {
        double next_value = dist_continuous(mt);

        double p_accept = 1.0; // Accept any move from zero-probability state
        double pdf_cur = pdf(cur_value);
        if (pdf_cur > 0.0) {
            p_accept = std::min(1.0, pdf(next_value) / pdf_cur);
        }
        if (dist_prob(mt) <= p_accept) {
            cur_value = next_value;
        }
    }
}

template <typename T>
McmcSampler<T>::McmcSampler(std::vector<T> &values, std::vector<double> &probs)
    : McmcSampler(values, probs, std::uniform_int_distribution<int>(0, static_cast<int>(values.size()) - 1)(mt))
{}

template <typename T>
McmcSampler<T>::McmcSampler(std::vector<T> &values, std::vector<double> &probs, int init_idx, int n_iterations_init)
    : values(values),
      probs(probs),
      dist_discrete(0, static_cast<int>(values.size()) - 1),
      dist_prob(0.0, 1.0),
      sample(&McmcSampler<T>::sample_discrete)
{
    cur_idx = init_idx;

    for (int i = 0; i < n_iterations_init; i++) { 
        int next_idx = dist_discrete(mt);
        double p_accept = std::min(1.0, probs[next_idx] / probs[cur_idx]);
        if (dist_prob(mt) <= p_accept) {
            cur_idx = next_idx;
        }
    }
}

template <typename T>
double McmcSampler<T>::sample_continuous()
{
    double next_value = dist_continuous(mt);

    double p_accept = 1.0; // Accept any move from zero-probability state
    double pdf_cur = pdf(cur_value);
    if (pdf_cur > 0.0) {
        p_accept = std::min(1.0, pdf(next_value) / pdf_cur);
    }
    if (dist_prob(mt) <= p_accept) {
       cur_value = next_value;
    }

    return cur_value;
}

template <typename T>
T McmcSampler<T>::sample_discrete()
{
    int next_idx = dist_discrete(mt);

    double p_accept = std::min(1.0, probs->get()[next_idx] / probs->get()[cur_idx]);
    if (dist_prob(mt) <= p_accept) {
       cur_idx = next_idx;
    }

    return values->get()[cur_idx];
}