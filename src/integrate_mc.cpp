/*
Implementation of the Monte Carlo methods with Vegas optimization.
*/

#include "../include/integrate_mc.hpp"
#include "../include/sample.hpp"

#include <random>
#include <cmath>
#include <ctime>
#include <vector>
#include <iostream>

Result integrate_MC(
    double lower,
    double upper,
    const std::function<double(double)> &f,
    int n_points,
    int n_bins,
    int n_iterations)
{
    struct Box {
        long points;
        double l;
        double u;
        double cur_integral;
    };
    std::vector<Box> bins(n_bins);
    
    double bin_size = (upper-lower) / n_bins;
    
    std::mt19937 mt(time(nullptr));
    std::uniform_real_distribution<double> dist(0, bin_size);
    
    double l = lower;
    for (int i = 0; i < n_bins; i++, l += bin_size) {
        double u = std::min(upper, l + bin_size);
        bins[i] = {n_points / n_bins, l, u};
    }

    double error_sum = 0;
    double result_sum = 0;

    for (int i = 0; i < n_iterations; i++) {
        double int_sum = 0;

        for (auto& bin: bins) {
            double sum = 0;
            double mean = 0;
            double m2 = 0;
            
            int points = bin.points;
            for (int k = 1; k <= points; k++) {
                double x = dist(mt) + bin.l;
                double y = f(x);
                sum += y;

                // Welford's variance
                double old_mean = mean;
                mean += (y - mean) / k;
                m2 += (y - old_mean) * (y - mean);
            }
            bin.cur_integral = bin_size * sum / points;
            int_sum += bin.cur_integral;
            error_sum += bin_size * bin_size * m2 / (points * (points-1));
        }

        for (auto& bin: bins) {
            double contribution = fabs(bin.cur_integral / int_sum);
            bin.points = std::max(2, int(n_points * contribution)); // Ensure at least 2 points per bin
        }
        result_sum += int_sum;
    }
    return {result_sum / n_iterations, sqrt(error_sum) / n_iterations};
}

Result integrate_MC_ndim(
    const std::vector<double> &lower,
    const std::vector<double> &upper,
    const std::function<double(const std::vector<double> &)> &f,
    int n_bins,
    int n_points)
{
    int dim = lower.size();
   
    std::mt19937 mt(time(nullptr));
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    std::uniform_int_distribution<int> bin_dist(0, n_bins - 1);

    std::vector<std::vector<double>> bin_distributions;
    std::vector<McmcSampler> bin_samplers;

    std::vector<double> bin_sizes(dim);

    std::vector<double> sampler_values(n_bins);
    for (int i = 0; i < n_bins; i++) {
      sampler_values[i] = i;
    }

    for(int i = 0; i < dim; i++) {
      // initialize to 1 so no box will have 0 probability
      bin_distributions.emplace_back(n_bins, 1.0);
    }

    double range = 1.0;

    for(int i = 0; i < dim; i++) {
      bin_samplers.emplace_back(sampler_values, bin_distributions[i]);
      bin_sizes[i] = (upper[i] - lower[i]) / n_bins;
      range *= (upper[i] - lower[i]);
    }

    std::vector<double> input(dim); 
     
    std::vector<int> chosen_bin(dim, 0);

    double sum = 0.0;
    double unweighted_sum = static_cast<double>(n_bins);
    double mean = 0.0;
    double m2 = 0.0;

    int burn_in_size = std::min(n_bins * 100, n_points);
    for (int i = 1; i <= n_points; i++) {
        double pdf = 1.0;
        if (i > burn_in_size) {
          for (int j = 0; j < dim; j++) {
            int bin = bin_samplers[j]();
            chosen_bin[j] = bin;
            input[j] = lower[j] + bin_sizes[j] * (bin + dist(mt));

            pdf *= (double)n_bins * bin_distributions[j][bin] / (unweighted_sum);
          }
        }
        else{
          for(int j = 0; j < dim; j++) {
            int bin = bin_dist(mt);
            chosen_bin[j] = bin;
            input[j] = lower[j] + bin_sizes[j] * (bin + dist(mt));
          }
        }

        double y = f(input) / pdf;
        sum += y;

        if (i <= burn_in_size) {
          unweighted_sum += y; 
          for(int j = 0; j < dim; j++) {
            bin_distributions[j][chosen_bin[j]] += y;
          }
        }

        // Welford's variance
        double old_mean = mean;
        mean += (y - mean) / i;
        m2 += (y - old_mean) * (y - mean);
    }
    double variance = m2  / (n_points - 1);
    double error = range * std::sqrt(variance / n_points);

    return {sum * range / n_points, error};
}

Result integrate_MC_dist(
    double lower,
    double upper,
    const std::function<double(double)> &f,
    const std::function<double(double)> &pdf,
    int n_points)
{
    McmcSampler sampler(lower, upper, pdf);

    std::vector<double> f_values;
    f_values.reserve(n_points);

    double mean = 0;
    double m2 = 0;

    for (int i = 1; i <= n_points; i++) {
        double x = sampler();
        double y = f(x) / pdf(x);

        f_values.push_back(y);

        double old_mean = mean;
        mean += (y - mean) / i;
        m2 += (y - old_mean) * (y - mean);
    }

    double var = m2 / (n_points - 1);

    // Computing autocorrelation time
    int max_lag = std::min(1000, n_points / 2);
    double tau_int = 1.0;

    for (int t = 1; t < max_lag; t++) {
        double autocov = 0.0;
        for (int i = 0; i < n_points - t; i++) {
            autocov += (f_values[i] - mean) * (f_values[i + t] - mean);
        }
        autocov /= (n_points - t);
        if (autocov <= 0) { // Gets too noisy, so stop here
            break;
        }

        tau_int += 2.0 * autocov / var;
    }

    double error = std::sqrt(var * tau_int / n_points);

    return {mean, error};
}