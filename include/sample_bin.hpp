#pragma once
#include "sample_mcmc_ndim.hpp"
#include <random>
#include <ctime>
#include <vector>

class BinSampler{
  public: 

  BinSampler(std::vector<std::vector<double>>& probs, std::vector<double>& lower, std::vector<double>& upper);

  std::vector<double> operator()()
  {
    return sample();
  }
  private:
    int n_bins; 
    int n_dims;

    std::vector<double>& sample();
    std::vector<double> &lower;
    std::vector<double> &upper;
    std::vector<std::vector<double>>& probs;

    static std::mt19937 mt;

    std::vector<std::vector<int>> bin_indices;

    McmcSampler<std::vector<int>> bin_sampler;
    std::vector<std::vector<int>> init_bin_indices();

    std::vector<double> cur_value;

    std::uniform_real_distribution<double> dist;
};

inline std::mt19937 BinSampler::mt{std::random_device{}()};

std::vector<std::vector<int>> BinSampler::init_bin_indices() {
    std::vector<int> indices(n_bins);
    std::iota(indices.begin(), indices.end(), 0);
    return std::vector<std::vector<int>>(n_dims, indices);
}

// For now, we assume every dimension has the same number of bins
BinSampler::BinSampler(std::vector<std::vector<double>>& probs, std::vector<double>& lower, std::vector<double>& upper)
  : n_bins(probs[0].size()),
    n_dims(probs.size()),
    probs(probs),
    lower(lower),
    upper(upper),
    bin_indices(init_bin_indices()),
    cur_value(n_dims),
    bin_sampler(bin_indices, probs),
    dist(0.0, 1.0)
{}

std::vector<double>& BinSampler::sample(){
  const std::vector<int>& cur_bins = bin_sampler();

  for(int i=0; i<n_dims; i++){
    double bin_width = (upper[i] - lower[i]) / n_bins;
    cur_value[i] = lower[i] + cur_bins[i] * bin_width + dist(mt) * bin_width;
  }
  return cur_value;
}
