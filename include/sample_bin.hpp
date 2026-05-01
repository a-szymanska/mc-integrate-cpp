#pragma once
#include "sample_mcmc_ndim.hpp"
#include <random>
#include <ctime>
#include <vector>


constexpr static int burnInSize = 10;

class BinSampler{
  public: 

  BinSampler(std::vector<std::vector<double>> probs, std::vector<double>& lower, std::vector<double>& upper);

  BinSampler(const std::function<double(const std::vector<double> &)> &f, int n_bins, std::vector<double>& lower, std::vector<double>& upper, int burn_in_size);

  std::vector<double> operator()()
  {
    return sample();
  }
  private:
    int n_bins; 
    int n_dims;

    std::uniform_real_distribution<double> dist;
    std::vector<double>& sample();
    std::vector<double> &lower;
    std::vector<double> &upper;
    std::vector<std::vector<double>> probs;

    static std::mt19937 mt;

    std::vector<std::vector<int>> bin_indices;

    McmcSampler<std::vector<int>> bin_sampler;
    std::vector<std::vector<int>> init_bin_indices();
    std::vector<std::vector<double>> estimate_distribution(const std::function<double(const std::vector<double> &)> &f, int burn_in_size = burnInSize);

    std::vector<double> cur_value;

};

inline std::mt19937 BinSampler::mt{std::random_device{}()};


std::vector<std::vector<double>> estimate_distribution(const std::function<double(const std::vector<double> &)> &f, int burn_in_size = burnInSize);
std::vector<std::vector<int>> BinSampler::init_bin_indices() {
    std::vector<int> indices(n_bins);
    std::iota(indices.begin(), indices.end(), 0);
    return std::vector<std::vector<int>>(n_dims, indices);
}

// For now, we assume every dimension has the same number of bins
BinSampler::BinSampler(std::vector<std::vector<double>> probs, std::vector<double>& lower, std::vector<double>& upper)
  : n_bins(probs[0].size()),
    n_dims(probs.size()),
    probs(probs),
    lower(lower),
    upper(upper),
    bin_indices(init_bin_indices()),
    cur_value(n_dims),
    bin_sampler(bin_indices, this->probs),
    dist(0.0, 1.0)
{}

std::vector<std::vector<double>> BinSampler::estimate_distribution(const std::function<double(const std::vector<double> &)> &f, int burn_in_size){
    std::vector<std::vector<double>> bin_probs(n_dims, std::vector<double>(n_bins, 0.0));
    std::vector<double> input(n_dims);
    for (int i = 0; i < n_dims; i++) {
        double burn_in_sum = 0.0;
        for (int j = 0; j < n_bins; j++) {
            for (int k = 0; k < burn_in_size; k++) {
                for (int l = 0; l < n_dims; l++) {
                  input[l] = lower[l] + (upper[l]-lower[l]) * dist(mt);
                }
                input[i] = lower[i] + (upper[i] - lower[i]) * (j + dist(mt));

                double y = std::fabs(f(input));
                bin_probs[i][j] += y;
            }
            burn_in_sum += bin_probs[i][j];
        }

        // Normalise the distribution
        for (int j = 0; j < n_bins; j++) {
            bin_probs[i][j] /= burn_in_sum;
        }
    }
    return bin_probs;
}

// estimating the distribution
BinSampler::BinSampler(const std::function<double(const std::vector<double> &)> &f, int n_bins, std::vector<double>& lower, std::vector<double>& upper, int burn_in_size = burnInSize)
  : n_bins(n_bins),
    n_dims(lower.size()),
    lower(lower),
    upper(upper),
    bin_indices(init_bin_indices()),
    cur_value(n_dims),
    dist(0.0, 1.0),
    probs(estimate_distribution(f, burn_in_size)),
    bin_sampler(bin_indices, probs)
{}

std::vector<double>& BinSampler::sample(){
  const std::vector<int>& cur_bins = bin_sampler();

  for(int i=0; i<n_dims; i++){
    double bin_width = (upper[i] - lower[i]) / n_bins;
    cur_value[i] = lower[i] + cur_bins[i] * bin_width + dist(mt) * bin_width;
  }
  return cur_value;
}
