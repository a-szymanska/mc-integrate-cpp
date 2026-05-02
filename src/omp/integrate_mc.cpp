#include "../include/integrate_mc.hpp"

#include <random>
#include <cmath>
#include <ctime>
#include <omp.h>

Result integrate_mc(
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
    
    double l = lower;
    for (int i = 0; i < n_boxes; i++, l += bin_size) {
        double u = std::min(upper, l + bin_size);
        bins[i] = {n_points / n_boxes, l, u};
    }

    double error_sum = 0;
    double result_sum = 0;

    for (int i = 0; i < n_iterations; i++) {
        double int_sum = 0;
        #pragma omp parallel
        {
            #pragma omp for reduction(+:int_sum,error_sum)
            for (int j=0; j<bins.size(); j++) {
                std::mt19937 local_mt(time(nullptr) + omp_get_thread_num());
                std::uniform_real_distribution<double> dist(bins[j].l, bins[j].l+bin_size);
                double sum = 0;
                double mean = 0;
                double m2 = 0;
                
                int points = bins[j].points;
                for (int k = 1; k <= points; k++) {
                    double x = dist(local_mt);
                    double y = f(x);
                    sum += y;

                    // Welford's variance
                    double old_mean = mean;
                    mean += (y-mean) / k;
                    m2 += (y-old_mean) * (y-mean);
                }
                bins[j].cur_integral = bin_size * sum / points;
                int_sum += bins[j].cur_integral;
                error_sum += bin_size * bin_size * m2 / (points * (points-1));
            }
        }

        for (auto& bin: bins) {
            double contribution = fabs(bin.cur_integral / int_sum);
            bin.points = std::max(2, int(n_points * contribution)); // Ensure at least 2 points per bin
        }
        result_sum += int_sum;
    }
    return {result_sum / n_iterations, sqrt(error_sum) / n_iterations};
}

template <typename Estimator>
Result integrate_mc_ndim(
    const std::vector<double> &lower,
    const std::vector<double> &upper,
    const std::function<double(const std::vector<double> &)> &f,
    int n_bins,
    int burn_in_size,
    int n_points)
{
    return {0, 0};
}

template <typename Estimator>
Result integrate_mc_highdim(
    const std::vector<double> &lower,
    const std::vector<double> &upper,
    const std::function<double(const std::vector<double> &)> &f,
    int n_bins,
    int burn_in_size,
    int n_points)
{
    return {0, 0};
}

template <typename Estimator>
Result integrate_mc_dist(
    double lower,
    double upper,
    const std::function<double(double)> &f,
    const std::function<double(double)> &pdf,
    int n_points)
{
    return {0, 0};
}
