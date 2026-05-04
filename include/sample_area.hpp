#pragma once
#include "sample_mcmc.hpp"
#include <random>
#include <numeric>
#include <cmath>
#include <vector>
#include <functional>

constexpr static int burnInSizeArea = 10;

class AreaSampler {
  public:

    AreaSampler(std::vector<double> probs, const std::vector<double>& lower,
                const std::vector<double>& upper, int n_bins);

    AreaSampler(const std::function<double(const std::vector<double>&)>& f, int n_bins,
                const std::vector<double>& lower, const std::vector<double>& upper,
                int burn_in_size = burnInSizeArea);

    std::vector<double> operator()() { return sample(); }

    double get_pdf();

  private:
    int n_bins;
    int n_dims;
    int n_areas;

    std::uniform_real_distribution<double> dist;
    std::vector<double>& sample();

    const std::vector<double>& lower;
    const std::vector<double>& upper;
    std::vector<double> bin_sizes;

    std::vector<double> area_probs;
    std::vector<std::vector<int>> areas; 
    std::vector<int> area_indices;

    McmcSampler<int> area_sampler;

    static std::mt19937 mt;

    std::vector<double> cur_value;
    double cur_pdf;
    double range;

    double init_range();
    std::vector<std::vector<int>> init_areas();
    std::vector<double> estimate_distribution(
        const std::function<double(const std::vector<double>&)>& f, int burn_in_size);
};

inline std::mt19937 AreaSampler::mt{std::random_device{}()};

double AreaSampler::init_range() {
    range = 1.0;
    for (int i = 0; i < n_dims; i++)
        range *= bin_sizes[i];
    return range;
}

std::vector<std::vector<int>> AreaSampler::init_areas() {
    std::vector<std::vector<int>> result(n_areas, std::vector<int>(n_dims));
    for (int i = 0; i < n_areas; i++) {
        int temp = i;
        for (int dim = n_dims - 1; dim >= 0; dim--) {
            result[i][dim] = temp % n_bins;
            temp /= n_bins;
        }
    }
    return result;
}

std::vector<double> AreaSampler::estimate_distribution(
    const std::function<double(const std::vector<double>&)>& f, int burn_in_size)
{
    std::vector<double> probs(n_areas, 0.0);
    std::vector<double> input(n_dims);
    double total = 0.0;

    for (int i = 0; i < n_areas; i++) {
        for (int j = 0; j < burn_in_size; j++) {
            for (int k = 0; k < n_dims; k++)
                input[k] = lower[k] + bin_sizes[k] * (areas[i][k] + dist(mt));
            probs[i] += std::fabs(f(input));
        }
        total += probs[i];
    }

    if (total > 0.0)
        for (auto& p : probs) p /= total;

    return probs;
}


AreaSampler::AreaSampler(std::vector<double> probs, const std::vector<double>& lower,
                         const std::vector<double>& upper, int n_bins)
    : n_bins(n_bins),
      n_dims(lower.size()),
      n_areas(static_cast<int>(std::round(std::pow(n_bins, lower.size())))),
      lower(lower),
      upper(upper),
      bin_sizes([&]{ std::vector<double> s(lower.size());
                     for (int i=0;i<(int)lower.size();i++) s[i]=(upper[i]-lower[i])/n_bins;
                     return s; }()),
      area_probs(std::move(probs)),
      areas(init_areas()),
      area_indices([&]{ std::vector<int> v(n_areas); std::iota(v.begin(),v.end(),0); return v; }()),
      cur_value(lower.size()),
      area_sampler(area_indices, area_probs),
      dist(0.0, 1.0),
      cur_pdf(1.0),
      range(init_range())
{}


AreaSampler::AreaSampler(const std::function<double(const std::vector<double>&)>& f, int n_bins,
                         const std::vector<double>& lower, const std::vector<double>& upper,
                         int burn_in_size)
    : n_bins(n_bins),
      n_dims(lower.size()),
      n_areas(static_cast<int>(std::round(std::pow(n_bins, lower.size())))),
      lower(lower),
      upper(upper),
      bin_sizes([&]{ std::vector<double> s(lower.size());
                     for (int i=0;i<(int)lower.size();i++) s[i]=(upper[i]-lower[i])/n_bins;
                     return s; }()),
      areas(init_areas()),
      area_indices([&]{ std::vector<int> v(n_areas); std::iota(v.begin(),v.end(),0); return v; }()),
      cur_value(lower.size()),
      dist(0.0, 1.0),
      area_probs(estimate_distribution(f, burn_in_size)),
      area_sampler(area_indices, area_probs),
      cur_pdf(1.0),
      range(init_range())
{}



std::vector<double>& AreaSampler::sample() {
    int area = area_sampler();
    for (int i = 0; i < n_dims; i++)
        cur_value[i] = lower[i] + bin_sizes[i] * (areas[area][i] + dist(mt));
    return cur_value;
}

double AreaSampler::get_pdf() {
    return area_sampler.get_pdf() / range;
}
