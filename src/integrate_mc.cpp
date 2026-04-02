/*
Implementation of the Monte Carlo methods with Vegas optimization.
*/

#include "../include/integrate_mc.hpp"
#include "../include/sample_mcmc.hpp"
#include "../include/utils.hpp"
#include <random>
#include <cmath>
#include <ctime>
#include <vector>

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
    int burn_in_size,
    int n_points)
{
  int n_dims = lower.size();
  int n_areas = std::pow(n_bins, n_dims);

  std::mt19937 mt(time(nullptr));
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  std::vector<double> bin_sizes(n_dims);

  double range = 1.0;
  for(int i = 0; i < n_dims; i++) {
    bin_sizes[i] = (upper[i] - lower[i]) / n_bins;
    range *= (upper[i] - lower[i]);
  }
  range/=n_areas;

  std::vector<std::vector<int>> areas(n_areas);
  std::vector<int> areas_indices(n_areas);

  for (int i = 0; i < n_areas; i++) {
    std::vector<int> combination(n_dims);
    int temp = i;
    for (int dim = n_dims - 1; dim >= 0; dim--)
    {        
      combination[dim] = temp % n_bins;
      temp /= n_bins;
    }
    areas[i] = combination;
    areas_indices[i]=i;
  }

  //generate bin_distribution, sample burn_in_size points from each area
  std::vector<double> area_dist(n_areas, 0.0);
  std::vector<double> input(n_dims);

  double burn_in_sum=0;
  for(int i=0; i<n_areas; i++){
    for(int j=0; j<burn_in_size; j++){
      for(int k =0; k<n_dims; k++){
        input[k]= lower[k] + bin_sizes[k] * (areas[i][k] + dist(mt));
      }
      double y = abs(f(input));
      area_dist[i]+=y;
    }
    burn_in_sum+=area_dist[i];
  }

  //normalise the distribution
  for(int i=0; i<n_areas; i++){
    area_dist[i]/=burn_in_sum;
  }

  //use the estimated distribution to calculate the integral
  std::vector<double> f_values;
  f_values.reserve(n_points);
  double mean = 0;
  double m2 = 0;

  McmcSampler area_sampler(areas_indices, area_dist);

  int max_lag = std::min(1000, n_points / 2); 
  int buffer_head = 0;
  std::vector<double> buffer(max_lag, 0.0);
  std::vector<double> autocov(max_lag, 0.0);
  for(int i=1; i<= n_points; i++){
    int area = area_sampler();
    for(int j =0; j<n_dims; j++){
      input[j]= lower[j] + bin_sizes[j] * (areas[area][j] + dist(mt));
    }
    double y = f(input)*range/area_dist[area]; 
    f_values.push_back(y);

    double old_mean = mean;
    mean += (y - mean) / i;
    m2 += (y - old_mean) * (y - mean);

    double y_mean = y - mean;
    for(int t=1; t<std::min(max_lag, i); t++){
      int pos = (buffer_head - t + max_lag) % max_lag; 
      autocov[t] += y_mean * buffer[buffer_head];
    }
    buffer[buffer_head] = y_mean;
    buffer_head = (buffer_head + 1) % max_lag;
  }
  
  double var = m2 / (n_points - 1);
 
  double error = compute_corellated_error(autocov,var, n_points); 
  return {mean, error};
}

Result integrate_MC_highdim(
    const std::vector<double> &lower,
    const std::vector<double> &upper,
    const std::function<double(const std::vector<double> &)> &f,
    int n_bins,
    int burn_in_size,
    int n_points)
{
    int n_dim = lower.size();
    std::mt19937 mt(time(nullptr));
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    std::uniform_int_distribution<int> bin_dist(0, n_bins - 1);

    std::vector<std::vector<double>> bin_distributions;
    std::vector<McmcSampler<int>> bin_samplers;

    std::vector<double> bin_sizes(n_dim);

    std::vector<int> sampler_values(n_bins);
    for (int i = 0; i < n_bins; i++) {
      sampler_values[i] = i;
    }

    for(int i = 0; i < n_dim; i++) {
      bin_distributions.emplace_back(n_bins, 0.0);
    }

    double range = 1.0;

    for(int i = 0; i < n_dim; i++) {
      bin_samplers.emplace_back(sampler_values, bin_distributions[i]);
      bin_sizes[i] = (upper[i] - lower[i]) / n_bins;
      range *= (upper[i] - lower[i]);
    }
    range/=std::pow(n_bins, n_dim);

    std::vector<double> input(n_dim);

    // estimate the distribution
    for(int i=0; i<n_dim; i++){
      double burn_in_sum = 0.0;
      for(int j=0; j<n_bins; j++){
        for(int k=0; k<burn_in_size; k++){
          for(int l=0; l<n_dim; l++){
            input[l] = lower[l] + bin_sizes[l]*n_bins * dist(mt);
          }
          input[i] = lower[i] + bin_sizes[i] * (j + dist(mt));

          double y = abs(f(input));
          bin_distributions[i][j]+=y;
        }
        burn_in_sum+=bin_distributions[i][j];
      }

      // normalise the distribution
      for(int j=0; j<n_bins; j++){
        bin_distributions[i][j]/=burn_in_sum;
      }
    }

    //use the estimated distribution to calculate the integral
    std::vector<double> f_values;
    f_values.reserve(n_points);
    double mean = 0;
    double m2 = 0;

    int max_lag = std::min(1000, n_points / 2); 
    int buffer_head = 0;
    std::vector<double> buffer(max_lag, 0.0);
    std::vector<double> autocov(max_lag, 0.0);
    for(int i=1; i<= n_points; i++){
      double pdf = 1.0;
      for (int j = 0; j < n_dim; j++) {
        int bin = bin_samplers[j]();
        input[j] = lower[j] + bin_sizes[j] * (bin + dist(mt));
        pdf *= bin_distributions[j][bin];
      }

      double y = f(input)*range/pdf; 
      f_values.push_back(y);

      double old_mean = mean;
      mean += (y - mean) / i;
      m2 += (y - old_mean) * (y - mean);

      double y_mean = y - mean;
      for(int t=1; t<std::min(max_lag, i); t++){
        int pos = (buffer_head - t + max_lag) % max_lag; 
        autocov[t] += y_mean * buffer[buffer_head];
      }
      buffer[buffer_head] = y_mean;
      buffer_head = (buffer_head + 1) % max_lag;
    }
    
    double var = m2 / (n_points - 1);
   
    double error = compute_corellated_error(autocov,var, n_points); 
    return {mean, error};
}

Result integrate_MC_dist(
    double lower,
    double upper,
    const std::function<double(double)> &f,
    const std::function<double(double)> &pdf,
    int n_points)
{
    McmcSampler<> sampler(lower, upper, pdf);

    std::vector<double> f_values;
    f_values.reserve(n_points);

    double mean = 0;
    double m2 = 0;

    int max_lag = std::min(1000, n_points / 2); 
    int buffer_head = 0;
    std::vector<double> buffer(max_lag, 0.0);
    std::vector<double> autocov(max_lag, 0.0);

    for (int i = 1; i <= n_points; i++) {
        double x = sampler();
        double y = f(x) / pdf(x);


        double old_mean = mean;
        mean += (y - mean) / i;
        m2 += (y - old_mean) * (y - mean);

        double y_mean = y - mean;
        for(int t=1; t<std::min(max_lag, i); t++){
          int pos = (buffer_head - t + max_lag) % max_lag; 
          autocov[t] += y_mean * buffer[buffer_head];
        }
        buffer[buffer_head] = y_mean;
        buffer_head = (buffer_head + 1) % max_lag;
    }

    double var = m2 / (n_points - 1);

    double error = compute_corellated_error(autocov,var, n_points); 
    return {mean, error};
}
