/*
Implementation of Monte Carlo and trapezoidal methods for numerical integration,
with Vegas optimization for improved convergence and Welford's algorithm for error estimation.
*/

#pragma once

#include <vector>
#include <functional>
#include <random>
#include <cmath>

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
    int n_points);

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
    std::mt19937 mt(time(nullptr));
    std::vector<box> bins(n_boxes);

    double bin_size=(upper-lower)/n_boxes;
    double bin_size_sq=bin_size*bin_size;

    std::uniform_real_distribution<double> dist(0, bin_size);
    double cur = lower;

    for(int i=0; i<n_boxes; i++){
        bins[i]={n_points/n_boxes, cur};
        cur+=bin_size;
    }

    double err_sum = 0;
    double result_sum = 0;

    for(int i=0; i<n_iterations; i++){
        double int_sum = 0;

        for(auto& bin: bins){
            double sum = 0;
            double mean=0;
            double m2=0;
            
            for(int k=1; k<=bin.points; k++){
                double x = dist(mt)+ bin.l;
                double y = f(x);
                sum+=y;

                //welford's variance
                double old_mean = mean;
                mean+=(y-mean)/k;
                m2+=(y-old_mean)*(y-mean);
            }
            bin.cur_integral = bin_size*sum/bin.points;
            int_sum+=bin.cur_integral;
            
            err_sum += bin_size_sq*m2/(bin.points*(bin.points-1));

        }

        for(auto& bin: bins){
            double contribution = fabs(bin.cur_integral/int_sum);
            bin.points=int(n_points*contribution);
            if(bin.points<2) bin.points=2;
        }
        result_sum+=int_sum;
    }
    return {result_sum/n_iterations, sqrt(err_sum)/n_iterations};
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
    int n_points)
{
    // TODO
    return {0.0, 0.0};
}
