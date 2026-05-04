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

template <typename Estimator>
Result integrate_MC_area(
    const std::vector<double> &lower,
    const std::vector<double> &upper,
    const std::function<double(const std::vector<double> &)> &f,
    int n_points,
    int n_bins,
    int burn_in_size
    )
{
    int n_dims = lower.size();
    int n_areas = std::pow(n_bins, n_dims);

    std::mt19937 mt(time(nullptr));
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    std::vector<double> bin_sizes(n_dims);

    double range = 1.0;
    for (int i = 0; i < n_dims; i++) {
        bin_sizes[i] = (upper[i] - lower[i]) / n_bins;
        range *= (upper[i] - lower[i]);
    }
    range /= n_areas;

    std::vector<std::vector<int>> areas(n_areas);
    std::vector<int> areas_indices(n_areas);

    for (int i = 0; i < n_areas; i++) {
        std::vector<int> combination(n_dims);
        int temp = i;
        for (int dim = n_dims - 1; dim >= 0; dim--) {
            combination[dim] = temp % n_bins;
            temp /= n_bins;
        }
        areas[i] = combination;
        areas_indices[i] = i;
    }

    // generate bin_distribution, sample burn_in_size points from each area
    std::vector<double> area_dist(n_areas, 0.0);
    std::vector<double> input(n_dims);

    double burn_in_sum = 0;
    for (int i = 0; i < n_areas; i++) {
        for (int j = 0; j < burn_in_size; j++) {
            for (int k = 0; k < n_dims; k++) {
                input[k] = lower[k] + bin_sizes[k] * (areas[i][k] + dist(mt));
            }
            double y = std::fabs(f(input));
            area_dist[i] += y;
        }
        burn_in_sum += area_dist[i];
    }

    // Normalise the distribution
    for (int i = 0; i < n_areas; i++) {
        area_dist[i] /= burn_in_sum;
    }

    // Use the estimated distribution to calculate the integral
    Estimator estimator(n_points);
    McmcSampler area_sampler(areas_indices, area_dist);

    for (int i = 1; i <= n_points; i++) {
        int area = area_sampler();
        for (int j = 0; j < n_dims; j++) {
            input[j] = lower[j] + bin_sizes[j] * (areas[area][j] + dist(mt));
        }
        double y = f(input) * range / area_dist[area];
        estimator.add_sample(y);
    }

    return {estimator.get_mean(), estimator.get_error()};
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
        double pdf = sampler.pdf();
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
