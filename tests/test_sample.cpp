#include "../include/sample_mcmc.hpp"
#include "../include/sample_mcmc_ndim.hpp"
#include "./utils.hpp"

#include <cassert>
#include <vector>

void test_sample_continuous()
{
    // An inverse parabola centered at 0.
    auto pdf = [](double x)
        { return 3/4 * (1 - x*x); };
        
    double lower = -1.0, upper = 1.0;
    McmcSampler<> sampler(lower, upper, pdf);
        
    int n_samples = 1000;
    double sum = 0.0;
    for (int i = 0; i < n_samples; i++) {
        sum += sampler();
    }

    double sample_mean = sum / n_samples;
    double expected_mean = 0.0;

    assert(relative_equal(sample_mean, expected_mean));
}

void test_sample_discrete()
{
    std::vector<int> values = {1, 2, 3, 4};
    std::vector<double> probs = {0.1, 0.2, 0.3, 0.4};

    McmcSampler<int> sampler(values, probs);

    int n_samples = 1000;
    double sum = 0.0;
    for (int i = 0; i < n_samples; i++) {
        sum += sampler();
    }

    double sample_mean = sum / n_samples;
    double expected_mean = 3.0;

    assert(relative_equal(sample_mean, expected_mean));
}

void test_sample_ndim_continuous()
{
    auto pdf = [](double x) { return 3.0 / 4.0 * (1 - x * x); };
    std::vector<std::function<double(double)>> pdfs = {pdf, pdf};

    std::vector<double> lower = {-1.0, -1.0};
    std::vector<double> upper = { 1.0,  1.0};
    McmcSampler<std::vector<double>> sampler(lower, upper, pdfs);

    int n_samples = 1000;
    std::vector<double> sum(2, 0.0);
    for (int i = 0; i < n_samples; i++) {
        auto s = sampler();
        sum[0] += s[0];
        sum[1] += s[1];
    }

    assert(relative_equal(sum[0] / n_samples, 0.0));
    assert(relative_equal(sum[1] / n_samples, 0.0));
}

void test_sample_ndim_discrete()
{
    std::vector<std::vector<double>> values = {{1.0, 2.0, 3.0, 4.0}, {10.0, 20.0, 30.0}};
    std::vector<std::vector<double>> probs  = {{0.1, 0.2, 0.3, 0.4}, { 0.2,  0.3,  0.5}};
    McmcSampler<std::vector<double>> sampler(values, probs);

    int n_samples = 1000;
    std::vector<double> sum(2, 0.0);
    for (int i = 0; i < n_samples; i++) {
        auto s = sampler();
        sum[0] += s[0];
        sum[1] += s[1];
    }

    assert(relative_equal(sum[0] / n_samples, 3.0));
    assert(relative_equal(sum[1] / n_samples, 23.0));
}

int main()
{
    test_sample_continuous();
    test_sample_discrete();
    test_sample_ndim_continuous();
    test_sample_discrete();
}
