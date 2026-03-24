#include "../include/sample_mcmc.hpp"
#include "./utils.hpp"

#include <cassert>
#include <cmath>
#include <vector>

void test_sample_continuous()
{
    // An inverse parabola centered at 0.
    auto pdf = [](double x)
        { return 3/4 * (1 - x*x); };

    int n_samples = 1000;
    double sum = 0.0;

    McmcSampler<> sampler(-1.0, 1.0, pdf);
    for (int i = 0; i < n_samples; i++) {
        auto x = sampler();
        sum += x;
    }

    double sample_mean = sum / n_samples;
    double expected_mean = 0.0;

    assert(relative_equal(sample_mean, expected_mean));
}

void test_sample_discrete()
{
    std::vector<int> values = {1, 2, 3, 4};
    std::vector<double> probs = {0.1, 0.2, 0.3, 0.4};

    int n_samples = 1000;
    double sum = 0.0;

    McmcSampler<int> sampler(values, probs);
    for (int i = 0; i < n_samples; i++) {
        auto x = sampler();
        sum += x;
    }

    double sample_mean = sum / n_samples;
    double expected_mean = 3.0;

    assert(relative_equal(sample_mean, expected_mean));
}

int main()
{
    test_sample_continuous();
    test_sample_discrete();
}
