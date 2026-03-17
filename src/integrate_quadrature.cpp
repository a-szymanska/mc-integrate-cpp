/*
Implementation of the trapezoidal method and Simpson's 1/3 quadrature
with adaptive partition into boxes.
*/

#include "../include/integrate_quadrature.hpp"

#include <vector>
#include <cmath>

Result integrate_quadrature(
    double lower,
    double upper,
    const std::function<double(double)> &f,
    const std::function<double(double, double, double, double)> &rule,
    int n_boxes,
    int n_iterations = 1)
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
            double I_single = rule(width, fl, fm, fu);
            // Value of the box split into two
            double I_split = rule(width, fl, (fl + fm) / 2.0, fm) + rule(width, fm, (fm + fu) / 2.0, fu);
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

Result integrate_trapezoid(
    double lower,
    double upper,
    const std::function<double(double)> &f,
    int n_boxes,
    int n_iterations)
{
    auto trapezoid_rule = [](double width, double fl, double fm, double fu) {
        return width / 2.0 * (fl + fu);
    };
    return integrate_quadrature(lower, upper, f, trapezoid_rule, n_boxes, n_iterations);
}

Result integrate_simpson(
    double lower,
    double upper,
    const std::function<double(double)> &f,
    int n_boxes,
    int n_iterations)
{
    auto simpson_rule = [](double width, double fl, double fm, double fu) {
        return width / 6.0 * (fl + 4.0 * fm + fu);
    };
    return integrate_quadrature(lower, upper, f, simpson_rule, n_boxes, n_iterations);
}