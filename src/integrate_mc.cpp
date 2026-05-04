/*
Implementation of the Monte Carlo methods with Vegas optimization.
*/

#pragma once
#include "../include/integrate_mc.hpp"
#include "../include/sample_mcmc.hpp"
#include "sample_bin.hpp"
#include <random>
#include <cmath>
#include <ctime>
#include <vector>

void allocate_bins(std::vector<int> &bin_points, std::vector<double> &bin_contributions, double total_contribution, int n_points)
{
    int used_points = 0;
    for (int i = 0; i < bin_points.size(); i++) {
        int points = bin_contributions[i] / total_contribution;
        bin_points[i] = 2 + points;
        used_points += points;
    }

    // TODO: randomly allocate the leftovers
}

Result integrate_MC(
    double lower,
    double upper,
    const std::function<double(double)> &f,
    int n_points,
    int n_bins,
    int n_iterations)
{
    struct Box
    {
        double l;
        double u;
        double cur_integral;
    };
    std::vector<Box> bins(n_bins);
    std::vector<int> bin_points(n_bins);

    double bin_size = (upper - lower) / n_bins;
    double range = bin_size * bin_size;
    std::mt19937 mt(time(nullptr));
    std::uniform_real_distribution<double> dist(0, bin_size);

    std::vector<double> bin_contributions(n_bins, 1.0);
    double total_contribution = n_bins;

    // Every bin will have at minimm two points, we allocate the rest by contribution
    n_points -= 2 * n_bins;
    if (n_points < 0) {
        throw "not enough points";
    }

    double l = lower;
    for (int i = 0; i < n_bins; i++, l += bin_size) {
        double u = std::min(upper, l + bin_size);
        bins[i] = {l, u};
    }

    allocate_bins(bin_points, bin_contributions, total_contribution, n_points);

    double error_sum = 0;
    double result_sum = 0;

    for (int i = 0; i < n_iterations; i++) {
        double int_sum = 0;

        for (int j = 0; j < bins.size(); j++) {
            int points = bin_points[j];
            EstimatorNoAutocorrelations estimator(points);

            for (int k = 1; k <= points; k++) {
                double x = dist(mt) + bins[j].l;
                double y = f(x);
                estimator.add_sample(y);
            }
            bins[j].cur_integral = bin_size * estimator.get_mean();
            int_sum += bins[j].cur_integral;
            error_sum += range * estimator.get_variance() / points;
        }

        total_contribution = 0.0;
        for (int j = 0; j < bins.size(); j++) {
            bin_contributions[j] = std::fabs(bins[j].cur_integral / int_sum);
            total_contribution += bin_contributions[j];
        }
        allocate_bins(bin_points, bin_contributions, total_contribution, n_points);

        result_sum += int_sum;
    }
    return {result_sum / n_iterations, sqrt(error_sum) / n_iterations};
}

template <typename Estimator, typename Sampler>
Result integrate_MC_ndim(
    const std::vector<double> &lower,
    const std::vector<double> &upper,
    const std::function<double(const std::vector<double> &)> &f,
    int n_points,
    int n_bins,
    int burn_in_size
    )
{
    BinSampler sampler(f, n_bins, lower, upper, burn_in_size);
    Estimator estimator(n_points);

    for (int i = 1; i <= n_points; i++) {
        auto point = sampler();
        double pdf = sampler.get_pdf();
        double y = f(point) / pdf;

        estimator.add_sample(y);
    }

    return {estimator.get_mean(), estimator.get_error()};
}



template <typename Estimator>
Result integrate_MC_dist(
    double lower,
    double upper,
    const std::function<double(double)> &f,
    const std::function<double(double)> &pdf,
    int n_points)
{
    McmcSampler<> sampler(lower, upper, pdf);
    Estimator estimator(n_points);

    for (int i = 1; i <= n_points; i++) {
        double x = sampler();
        double y = f(x) / pdf(x);

        estimator.add_sample(y);
    }

    return {estimator.get_mean(), estimator.get_error()};
}
