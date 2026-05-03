#pragma once
#include "sample_mcmc_ndim.hpp"
#include <random>
#include <ctime>
#include <vector>


constexpr static int burnInSize = 10;

class BinSampler{
  public: 

  BinSampler(std::vector<std::vector<double>> probs, const std::vector<double>& lower, const std::vector<double>& upper);

  BinSampler(const std::function<double(const std::vector<double> &)> &f, int n_bins, const std::vector<double>& lower, const std::vector<double>& upper, int burn_in_size);

  std::vector<double> operator()()
  {
    return sample();
  }

  double pdf();

  private:
    int n_bins; 
    int n_dims;

    std::uniform_real_distribution<double> dist;
    std::vector<double>& sample();
    const std::vector<double> &lower;
    const std::vector<double> &upper;
    std::vector<std::vector<double>> probs;

    static std::mt19937 mt;

    std::vector<std::vector<int>> bin_indices;

    McmcSampler<std::vector<int>> bin_sampler;
    std::vector<std::vector<int>> init_bin_indices();
    std::vector<std::vector<double>> estimate_distribution(const std::function<double(const std::vector<double> &)> &f, int burn_in_size = burnInSize);

    std::vector<double> cur_value;
    double cur_pdf;
    double range;
    double init_range();
};

inline std::mt19937 BinSampler::mt{std::random_device{}()};


std::vector<std::vector<double>> estimate_distribution(const std::function<double(const std::vector<double> &)> &f, int burn_in_size = burnInSize);
std::vector<std::vector<int>> BinSampler::init_bin_indices() {
    std::vector<int> indices(n_bins);
    std::iota(indices.begin(), indices.end(), 0);
    return std::vector<std::vector<int>>(n_dims, indices);
}

double BinSampler::init_range(){
  range = 1.0;
  for (int i = 0; i < n_dims; i++) {
      range *= (upper[i] - lower[i]);
  }
  range /= std::pow(n_bins, n_dims);

  return range;
}

// For now, we assume every dimension has the same number of bins
BinSampler::BinSampler(std::vector<std::vector<double>> probs, const std::vector<double>& lower, const std::vector<double>& upper)
  : n_bins(probs[0].size()),
    n_dims(probs.size()),
    probs(probs),
    lower(lower),
    upper(upper),
    bin_indices(init_bin_indices()),
    cur_value(n_dims),
    bin_sampler(bin_indices, this->probs),
    dist(0.0, 1.0),
    cur_pdf(1.0),
    range(init_range())
{}

std::vector<std::vector<double>> BinSampler::estimate_distribution(
    const std::function<double(const std::vector<double> &)> &f, int burn_in_size)
{
    std::vector<std::vector<double>> bin_probs(n_dims, std::vector<double>(n_bins, 0.0));
    std::vector<double> sums(n_dims, 0.0);
    std::vector<double> input(n_dims);

    int total_samples = burn_in_size * n_bins; // same evaluation budget as before

    for (int k = 0; k < total_samples; k++) {
        // One random point covering the full space
        for (int l = 0; l < n_dims; l++) {
            input[l] = lower[l] + (upper[l] - lower[l]) * dist(mt);
        }

        double y = std::fabs(f(input));

        // Accumulate into every dimension's marginal bin
        for (int i = 0; i < n_dims; i++) {
            int bin = static_cast<int>(
                (input[i] - lower[i]) / (upper[i] - lower[i]) * n_bins
            );
            bin = std::clamp(bin, 0, n_bins - 1);
            bin_probs[i][bin] += y;
            sums[i] += y;
        }
    }

    // Normalise each dimension independently
    for (int i = 0; i < n_dims; i++) {
        if (sums[i] > 0.0) {
            for (int j = 0; j < n_bins; j++) {
                bin_probs[i][j] /= sums[i];
            }
        }
    }

    return bin_probs;
}

// estimating the distribution
BinSampler::BinSampler(const std::function<double(const std::vector<double> &)> &f, int n_bins, const std::vector<double>& lower, const std::vector<double>& upper, int burn_in_size = burnInSize)
  : n_bins(n_bins),
    n_dims(lower.size()),
    lower(lower),
    upper(upper),
    bin_indices(init_bin_indices()),
    cur_value(n_dims),
    dist(0.0, 1.0),
    probs(estimate_distribution(f, burn_in_size)),
    bin_sampler(bin_indices, probs),
    cur_pdf(1.0),
    range(init_range())
{}

std::vector<double>& BinSampler::sample(){
  const std::vector<int>& cur_bins = bin_sampler();

  for(int i=0; i<n_dims; i++){
    double bin_width = (upper[i] - lower[i]) / n_bins;
    cur_value[i] = lower[i] + cur_bins[i] * bin_width + dist(mt) * bin_width;
  }
  return cur_value;
}

double BinSampler::pdf(){
  return bin_sampler.pdf()/range;
}
