/*
Implementation of Monte Carlo and trapezoidal methods for numerical integration,
with Vegas optimization for improved convergence and Welford's algorithm for error estimation.
*/

#pragma once

#include <vector>
#include <functional>
#include <random>
#include <cmath>
#include <ctime>

constexpr double kTolerance = 1e-6;

struct Result {
    double value;
    double error;
};

// ----- Monte Carlo integration (1-dimensional) -----

Result integrate_MC(
    double lower,
    double upper,
    const std::function<double(double)> &f,
    int n_points,
    int n_boxes,
    int n_iterations);

// ----- Monte Carlo integration (N-dimensional) -----

Result integrate_MC_ndim(
    const std::vector<double> &lower,
    const std::vector<double> &upper,
    const std::function<double(const std::vector<double> &)> &f,
    int n_points);

// ------ Monte Carlo with custom distribution ------

Result integrate_MC_dist(
    double lower,
    double upper,
    const std::function<double(double)> &f,
    const std::function<double(double)> &p,
    int n_points,
    int n_boxes,
    int n_iterations);

// -------------- Trapezoid integration --------------

Result integrate_trapezoid(
    double lower,
    double upper,
    const std::function<double(double)> &f,
    int n_boxes,
    int n_iterations);


Result integrate_MC(
    double lower,
    double upper,
    const std::function<double(double)> &f,
    int n_points,
    int n_boxes,
    int n_iterations)
{
    struct box{
        long points;
        long double l;
        long double cur_integral;
    };
    std::vector<box> bins(n_boxes);
    
    double bin_size = (upper-lower) / n_boxes;
    
    std::mt19937 mt(time(nullptr));
    std::uniform_real_distribution<double> dist(0, bin_size);
    double cur = lower;

    for (int i = 0; i < n_boxes; i++) {
        bins[i] = {n_points / n_boxes, cur};
        cur += bin_size;
    }

    double err_sum = 0;
    double result_sum = 0;

    for (int i = 0; i < n_iterations; i++) {
        double int_sum = 0;

        for (auto& bin: bins) {
            double sum = 0;
            double mean = 0;
            double m2 = 0;
            
            for (int k = 1; k <= bin.points; k++) {
                double x = dist(mt) + bin.l;
                double y = f(x);
                sum += y;

                // Welford's variance
                double old_mean = mean;
                mean += (y-mean) / k;
                m2 += (y-old_mean) * (y-mean);
            }
            bin.cur_integral = bin_size * sum / bin.points;
            int_sum += bin.cur_integral;
            
            err_sum += bin_size * bin_size * m2 / (bin.points * (bin.points-1));

        }

        for (auto& bin: bins) {
            double contribution = fabs(bin.cur_integral / int_sum);
            bin.points = std::max(2, int(n_points * contribution)); // ensure at least 2 points per bin
        }
        result_sum += int_sum;
    }
    return {result_sum / n_iterations, sqrt(err_sum) / n_iterations};
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
    int max_iterations = 10)
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

    for (int iter = 0; iter < max_iterations; iter++) {
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
            double I_split =
                (width / 2.0) * (fl + fm) / 2.0 +
                (width / 2.0) * (fm + fu) / 2.0;

            double diff = std::abs(I_split - I_single);

            if (diff > kTolerance && iter < max_iterations - 1) {
                new_bins.push_back({l, m});
                new_bins.push_back({m, u});
            } else {
                result_sum += I_split;

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