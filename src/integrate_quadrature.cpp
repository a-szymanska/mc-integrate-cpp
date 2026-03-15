/*
Implementation of the trapezoidal method and Simpson's 1/3 quadrature
with adaptive partition into boxes.
*/

#include "../include/integrate_quadrature.hpp"

#include <vector>
#include <cmath>


Result integrate_trapezoid(
    double lower,
    double upper,
    const std::function<double(double)> &f,
    int n_boxes,
    int n_iterations)
{
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

            if (diff > kBoxTolerance && iter < n_iterations - 1) {
                new_bins.push_back({l, m});
                new_bins.push_back({m, u});
            } else {
                result_sum += I_single;

                double var = getBoxVariance(fl, fm, fu);
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
    int n_iterations)
{
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

            if (diff > kBoxTolerance && iter < n_iterations - 1) {
                new_bins.push_back({l, m});
                new_bins.push_back({m, u});
            } else {
                result_sum += I_single;

                double var = getBoxVariance(fl, fm, fu);
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