/*
  mc_integrate_cpp.hpp - amalgamated version
  https://github.com/a-szymanska/mc-integrate-cpp
  Distributed under the MIT License
  Generated on sob 28 mar 22:38:18 2026 CET
*/

#pragma once

#include <cinttypes>
#include <cmath>
#include <ctime>
#include <functional>
#include <optional>
#include <random>
#include <type_traits>
#include <vector>

// ===== include/common.hpp =====
struct Result
{
    double value;
    double error;
};

// ===== include/integrate_mc.hpp =====
/*
Implementation of Monte Carlo methods for numerical integration,
with Vegas optimization and Welford's algorithm for error estimation.
*/





/*
The total number of sampled points is equal to n_points * n_iterations.
*/

/*
The returned error estimates the standard deviation of the integral estimate
based on the sample variance of the function in each bin.
*/

// ----- Monte Carlo integration (1-dimensional) -----

Result integrate_MC(
    double lower,
    double upper,
    const std::function<double(double)> &f,
    int n_points,
    int n_bins,
    int n_iterations);

// ----- Monte Carlo integration (N-dimensional) -----

Result integrate_MC_ndim(
    const std::vector<double> &lower,
    const std::vector<double> &upper,
    const std::function<double(const std::vector<double> &)> &f,
    int n_bins,
    int burn_in_size,
    int n_points);


// ----- Monte Carlo integration (N-dimensional) optimised for high dimension count -----

Result integrate_MC_hdim(
    const std::vector<double> &lower,
    const std::vector<double> &upper,
    const std::function<double(const std::vector<double> &)> &f,
    int n_bins,
    int burn_in_size,
    int n_points);

// ------ Monte Carlo with custom distribution ------

Result integrate_MC_dist(
    double lower,
    double upper,
    const std::function<double(double)> &f,
    const std::function<double(double)> &pdf,
    int n_points);

// ===== include/integrate_quadrature.hpp =====
/*
The quadrature methods for numerical integration with adaptive partition into boxes.
*/




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

// ===== include/sample_mcmc.hpp =====
/*
The sampling algorithms for importance Monte Carlo methods.
*/



// ------------ Metropolis-Hastings (MCMC) -----------

template <typename T = double>
class McmcSampler
{
public:
    // Continuous distribution constructor
    template <typename U = T, typename = std::enable_if_t<std::is_same<U, double>::value>>
    McmcSampler(double lower, double upper, std::function<double(double)> pdf);
    // Implemented only for T = double.

    // Discrete distribution constructor
    McmcSampler(std::vector<T> &values, std::vector<double> &probs);
    
    T operator()()
    {
        return (this->*sample)();
    }

private:
    static std::mt19937 mt;
    std::uniform_real_distribution<double> dist_prob;
    
    using sample_fn = T (McmcSampler<T>::*)();
    sample_fn sample;

    constexpr static int kNumIterationsInit = 1000;

    // ----------- Continuous case -----------

    double lower;
    double upper;
    std::function<double(double)> pdf;
    double cur_value;

    std::uniform_real_distribution<double> dist_continuous;
    double sample_continuous();

    // ----------- Discrete case -----------

    std::optional<std::reference_wrapper<std::vector<T>>> values;
    std::optional<std::reference_wrapper<std::vector<double>>> probs;
    int cur_idx;

    std::uniform_int_distribution<int> dist_discrete;
    T sample_discrete();
};


template <typename T>
std::mt19937 McmcSampler<T>::mt{std::random_device{}()};

template <typename T>
template <typename U, typename>
McmcSampler<T>::McmcSampler(double lower, double upper, std::function<double(double)> pdf)
    : lower(lower),
      upper(upper),
      pdf(pdf),
      dist_continuous(lower, upper),
      dist_prob(0.0, 1.0),
      sample(&McmcSampler<T>::sample_continuous)
{
    cur_value = dist_continuous(mt);

    for (int i = 0; i < kNumIterationsInit; i++) { 
        double next_value = dist_continuous(mt);

        double p_accept = 1.0; // Accept any move from zero-probability state
        double pdf_cur = pdf(cur_value);
        if (pdf_cur > 0.0) {
            p_accept = std::min(1.0, pdf(next_value) / pdf_cur);
        }
        if (dist_prob(mt) <= p_accept) {
        cur_value = next_value;
        }
    }
}

template <typename T>
McmcSampler<T>::McmcSampler(std::vector<T> &values, std::vector<double> &probs)
    : values(values),
      probs(probs),
      dist_discrete(0, static_cast<int>(values.size()) - 1),
      dist_prob(0.0, 1.0),
      sample(&McmcSampler<T>::sample_discrete)
{
    cur_idx = dist_discrete(mt);

    for (int i = 0; i < kNumIterationsInit; i++) { 
        int next_idx = dist_discrete(mt);
        double p_accept = std::min(1.0, probs[next_idx] / probs[cur_idx]);
        if (dist_prob(mt) <= p_accept) {
            cur_idx = next_idx;
        }
    }
}

template <typename T>
double McmcSampler<T>::sample_continuous()
{
    double next_value = dist_continuous(mt);

    double p_accept = 1.0; // Accept any move from zero-probability state
    double pdf_cur = pdf(cur_value);
    if (pdf_cur > 0.0) {
        p_accept = std::min(1.0, pdf(next_value) / pdf_cur);
    }
    if (dist_prob(mt) <= p_accept) {
       cur_value = next_value;
    }

    return cur_value;
}

template <typename T>
T McmcSampler<T>::sample_discrete()
{
    int next_idx = dist_discrete(mt);

    double p_accept = std::min(1.0, probs->get()[next_idx] / probs->get()[cur_idx]);
    if (dist_prob(mt) <= p_accept) {
       cur_idx = next_idx;
    }

    return values->get()[cur_idx];
}

// ===== src/integrate_mc.cpp =====
/*
Implementation of the Monte Carlo methods with Vegas optimization.
*/



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

Result integrate_MC_hdim(
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

// ===== src/integrate_quadrature.cpp =====
/*
Implementation of the trapezoidal method and Simpson's 1/3 quadrature
with adaptive partition into boxes.
*/



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

