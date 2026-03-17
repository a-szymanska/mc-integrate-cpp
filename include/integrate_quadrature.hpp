/*
The quadrature methods for numerical integration with adaptive partition into boxes.
*/

#pragma once

#include "common.hpp"

#include <functional>

constexpr double kBoxTolerance = 1e-6;

struct Box
{
    double l;
    double u;
};

/*
If n_iterations > 1, in each iteration every box is split into two sub-boxes
if this improves the integral estimate by more than kBoxTolerance.
The final number of boxes is at most n_boxes * 2^(n_iterations - 1).
*/

/*
The returned error estimates the standard deviation of the integral estimate
based on the variation of the function values within each bin.
*/

Result integrate_quadrature(
    double lower,
    double upper,
    const std::function<double(double)> &f,
    const std::function<double(double, double, double, double)> &rule,
    int n_boxes,
    int n_iterations);


// -------------- Trapezoid integration --------------

Result integrate_trapezoid(
    double lower,
    double upper,
    const std::function<double(double)> &f,
    int n_boxes,
    int n_iterations = 1);

// -------------- Simpson's integration --------------

Result integrate_simpson(
    double lower,
    double upper,
    const std::function<double(double)> &f,
    int n_boxes,
    int n_iterations = 1);


// ---------------------- Utils ----------------------

inline double getBoxVariance(double fl, double fm, double fu)
{
    double mean = (fl + fm + fu) / 3.0;
    double var =
         ((fl - mean) * (fl - mean)
        + (fm - mean) * (fm - mean)
        + (fu - mean) * (fu - mean)) / 2.0;
    return var;
}
