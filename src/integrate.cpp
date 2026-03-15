/*
Implementation of Monte Carlo methods with Vegas optimization.
*/

#include "../include/integrate.hpp"

#include <vector>
#include <functional>
#include <random>
#include <cmath>
#include <ctime>

#include <iostream>

Result integrate_MC(
    double lower,
    double upper,
    const std::function<double(double)> &f,
    int n_points,
    int n_boxes,
    int n_iterations)
{
    struct Box {
        long points;
        double l;
        double u;
        double cur_integral;
    };
    std::vector<Box> bins(n_boxes);
    
    double bin_size = (upper-lower) / n_boxes;
    
    std::mt19937 mt(time(nullptr));
    std::uniform_real_distribution<double> dist(0, bin_size);
    
    double l = lower;
    for (int i = 0; i < n_boxes; i++, l += bin_size) {
        double u = std::min(upper, l + bin_size);
        bins[i] = {n_points / n_boxes, l, u};
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
                mean += (y-mean) / k;
                m2 += (y-old_mean) * (y-mean);
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
    int n_points)
{
    // TODO
    return {0.0, 0.0};
}

Result integrate_MC_dist(
    double lower,
    double upper,
    const std::function<double(double)> &f,
    const std::function<double(double)> &p,
    int n_points,
    int n_boxes,
    int n_iterations)
{
    // TODO
    return {0.0, 0.0};
}

Result integrate_trapezoid(
    double lower,
    double upper,
    const std::function<double(double)> &f,
    int n_boxes,
    int n_iterations=10)
{
    struct Box {
        double l;
        double u;
    };

    std::vector<Box> bins;

    double bin_size = (upper - lower) / n_boxes;

    double l = lower;
    for (int i = 0; i < n_boxes; i++, l += bin_size) {
        double u = std::min(upper, l + bin_size);
        bins.push_back({l, u});
    }

    double result_sum = 0.0;
    double error_sum = 0.0;

    for (int iter = 0; iter < n_iterations; iter++) {
        std::vector<Box> new_bins;

        for (auto &bin : bins) {
            double l = bin.l, u = bin.u;
            double m = (l + u) / 2.0;

            double fl = f(l);
            double fm = f(m);
            double fu = f(u);

            double width = u - l;

            // Value of the current box
            double I_single = width * (fl + fu) / 2;
            // Value of the box split into two
            double I_split = width * ((fl + fm) + (fm + fu)) / 4.0;
            double diff = std::abs(I_split - I_single);

            if (diff > kTolerance && iter < n_iterations - 1) {
                new_bins.push_back({l, m});
                new_bins.push_back({m, u});
            } else {
                result_sum += I_single;

                // Variance
                double mean = (fl + fm + fu) / 3.0;
                double var =
                    ((fl - mean) * (fl - mean)
                    + (fm - mean) * (fm - mean)
                    + (fu - mean) * (fu - mean)) / 2.0;
                double I_var = width * width * var / 3.0;
                error_sum += I_var;
            }
        }

        if (new_bins.empty()) {
            break;
        }
        bins = std::move(new_bins);
    }
    return {result_sum, std::sqrt(error_sum)};
}

Result integrate_simpson(
    double lower,
    double upper,
    const std::function<double(double)> &f,
    int n_boxes,
    int n_iterations=10)
{
    struct Box {
        double l;
        double u;
    };

    std::vector<Box> bins;

    double bin_size = (upper - lower) / n_boxes;

    double l = lower;
    for (int i = 0; i < n_boxes; i++, l += bin_size) {
        double u = std::min(upper, l + bin_size);
        bins.push_back({l, u});
    }

    double result_sum = 0.0;
    double error_sum = 0.0;

    for (int iter = 0; iter < n_iterations; iter++) {
        std::vector<Box> new_bins;

        for (auto &bin : bins) {
            double l = bin.l, u = bin.u;
            double m = (l + u) / 2.0;

            double fl = f(l);
            double fm = f(m);
            double fu = f(u);

            double width = u - l;

            // Value of the current box
            double I_single = width / 6.0 * (fl + 4*fm + fu);
            // Value of the box split into two
            double I_split = width / 12.0 * ((fl + 2*(fl + fm) + fm) + (fm + 2*(fm + fu) + fu));

            double diff = std::abs(I_split - I_single);

            if (diff > kTolerance && iter < n_iterations - 1) {
                new_bins.push_back({l, m});
                new_bins.push_back({m, u});
            } else {
                result_sum += I_single;

                // Variance
                double mean = (fl + fm + fu) / 3.0;
                double var =
                    ((fl - mean) * (fl - mean)
                    + (fm - mean) * (fm - mean)
                    + (fu - mean) * (fu - mean)) / 2.0;
                double I_var = width * width * var / 3.0;
                error_sum += I_var;
            }
        }

        if (new_bins.empty()) {
            break;
        }
        bins = std::move(new_bins);
    }
    return {result_sum, std::sqrt(error_sum)};
}