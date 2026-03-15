/*
Implementation of the Monte Carlo methods with Vegas optimization.
*/

#include "../include/integrate_mc.hpp"

#include <random>
#include <cmath>
#include <ctime>

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
    int dim = lower.size();

    std::mt19937 mt(time(nullptr));
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    double sum = 0;
    double mean = 0;
    double m2 = 0;

    double range = 1;

    for(int i = 0; i < dim; i++){
      range *= (upper[i]-lower[i]);
    }

    std::vector<double> input(dim); 
  
    for (int i = 1; i <= n_points; i++) {
        for(int j = 0; j < dim; j++){
          input[j] = lower[j] + (upper[j] - lower[j]) * dist(mt);
        }

        double y = f(input);
        sum += y;

        // Welford's variance
        double old_mean = mean;
        mean += (y-mean) / i;
        m2 += (y-old_mean) * (y-mean);
    }
    double variance = m2  / (n_points - 1);
    double error = range * std::sqrt(variance / n_points);

    return {sum * range/ n_points, error};
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
