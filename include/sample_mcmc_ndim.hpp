/*
The multidimensional sampling algorithms for importance Monte Carlo methods.
*/

#pragma once

#include <random>
#include <vector>
#include <functional>
#include "sample_mcmc.hpp"

template <typename U>
class McmcSampler<std::vector<U>>{
  public:
  
    template <typename S = U, typename = std::enable_if_t<std::is_same<S, double>::value>>
    McmcSampler(std::vector<double>& lower, std::vector<double>& upper, std::vector<std::function<double(double)>>& pdfs);

    template <typename S = U, typename = std::enable_if_t<std::is_same<S, double>::value>>
    McmcSampler(std::vector<double>& lower, std::vector<double>& upper, std::vector<std::function<double(double)>>& pdfs, std::vector<double> init_value, int n_iterations_init = kNumIterationsInit);


    McmcSampler(std::vector<std::vector<U>> &values, std::vector<std::vector<double>> &probs);
    McmcSampler(std::vector<std::vector<U>> &values, std::vector<std::vector<double>> &probs, std::vector<int> init_idx, int n_iterations_init = kNumIterationsInit);

    std::vector<U> operator()()
    {
        return (this->*sample)();
    }

private:
    int n_dim;
    static std::mt19937 mt;
    std::uniform_real_distribution<double> dist_prob;

    using sample_fn = const std::vector<U>&(McmcSampler<std::vector<double>>::*)();
    sample_fn sample;

    std::vector<U> cur_value;

    // ----------- Continuous case -----------

    std::optional<std::reference_wrapper<std::vector<double>>> lower;
    std::optional<std::reference_wrapper<std::vector<double>>> upper;
    std::vector<std::function<double(double)>> pdfs;

    
    std::uniform_real_distribution<double> dist_continuous;
    const std::vector<double>& sample_continuous();
    double sample_continous_coordinate(int dim);
    std::vector<double> sample_continous_init();


    // ----------- Discrete case -----------

    std::optional<std::reference_wrapper<std::vector<std::vector<U>>>> values;
    std::optional<std::reference_wrapper<std::vector<std::vector<double>>>> probs;
    std::uniform_int_distribution<int> dist_discrete;
    std::vector<int> cur_idx;
    const std::vector<U>& sample_discrete();
    int sample_discrete_coordinate(int dim);
    std::vector<int> sample_discrete_init();
};


template <typename U>
inline std::mt19937 McmcSampler<std::vector<U>>::mt{std::random_device{}()};

template <>
std::vector<double> McmcSampler<std::vector<double>>::sample_continous_init(){
  std::vector<double> res(n_dim, 0);
  for(int i = 0; i<n_dim; i++){
    res[i] = sample_continous_coordinate(i);
  }
  return res;
}

template <typename U>
inline double McmcSampler<std::vector<U>>::sample_continous_coordinate(int dim)
{
  return lower->get()[dim] + (dist_continuous(mt) * (upper->get()[dim] - lower->get()[dim]));
}


template <typename U>
const std::vector<double>& McmcSampler<std::vector<U>>::sample_continuous()
{
    // we do a full sweep every time
    for(int i=0; i<n_dim; i++){
      double next_value = sample_continous_coordinate(i);

      double p_accept = 1.0; // Accept any move from zero-probability state
      double pdf_cur = pdfs[i](cur_value[i]);
      if (pdf_cur > 0.0) {
          p_accept = std::min(1.0, pdfs[i](next_value) / pdf_cur);
      }
      if (dist_prob(mt) <= p_accept) {
         cur_value[i] = next_value;
      }

    }
    return cur_value;
}


template <typename U>
template <typename S, typename>
McmcSampler<std::vector<U>>::McmcSampler(std::vector<double>& lower, std::vector<double>& upper, std::vector<std::function<double(double)>>& pdfs)
  :  n_dim(lower.size()),
    lower(lower),
    upper(upper),
    pdfs(pdfs),
    dist_continuous(0.0, 1.0),
    sample(&McmcSampler::sample_continuous),
    cur_value(sample_continous_init())
    {
      for (int i = 0; i < kNumIterationsInit; i++) {
        sample_continuous();
      }
    }

template <typename U>
template <typename S, typename>
McmcSampler<std::vector<U>>::McmcSampler(std::vector<double>& lower, std::vector<double>& upper, std::vector<std::function<double(double)>>& pdfs, std::vector<double> init_value, int n_iterations_init)
  : n_dim(lower.size()),
    lower(lower),
    upper(upper),
    pdfs(pdfs),
    dist_continuous(0.0, 1.0),
    cur_value(init_value),
    sample(&McmcSampler::sample_continuous)
    {
      for (int i = 0; i < n_iterations_init; i++) {
        sample_continuous();
      }
    }


template <typename U>
std::vector<int> McmcSampler<std::vector<U>>::sample_discrete_init(){
  std::vector<int> res(n_dim, 0);
  for(int i = 0; i<n_dim; i++){
    res[i] = sample_discrete_coordinate(i);
  }
  return res;
}


template <typename U>
inline int McmcSampler<std::vector<U>>::sample_discrete_coordinate(int dim)
{
  return dist_discrete(mt) % static_cast<int>(values->get()[dim].size());
}


template <typename U>
const std::vector<U>& McmcSampler<std::vector<U>>::sample_discrete()
{
    // we do a full sweep every time
    for(int i=0; i<n_dim; i++){
      int next_idx = sample_discrete_coordinate(i);        
      double p_accept = std::min(1.0, probs->get()[i][next_idx] / probs->get()[i][cur_idx[i]]);

      if (dist_prob(mt) <= p_accept) {
         cur_idx[i] = next_idx;
         cur_value[i] = values->get()[i][next_idx];
      }

    }
    return cur_value;
}


template <typename U>
McmcSampler<std::vector<U>>::McmcSampler(std::vector<std::vector<U>> &values, std::vector<std::vector<double>> &probs)
  : n_dim(probs.size()),
    values(values),
    probs(probs),
    dist_discrete(),
    dist_prob(0.0, 1.0),
    sample(&McmcSampler<std::vector<double>>::sample_discrete),
    cur_idx(sample_discrete_init())
{
  cur_value.resize(n_dim);
  for(int i = 0; i < n_dim; i++){
    cur_value[i]=values[i][cur_idx[i]];
  }

  for(int i = 0; i < kNumIterationsInit; i++) {
    sample_discrete();
  }
}


template <typename U>
McmcSampler<std::vector<U>>::McmcSampler(std::vector<std::vector<U>> &values, std::vector<std::vector<double>> &probs, std::vector<int> init_idx, int n_iterations_init)
  : n_dim(probs.size()),
    values(values),
    probs(probs),
    dist_discrete(),
    dist_prob(0.0, 1.0),
    sample(&McmcSampler<std::vector<double>>::sample_discrete),
    cur_idx(init_idx)
{
  cur_value.resize(n_dim);
  for(int i = 0; i < n_dim; i++){
    cur_value[i]=values[i][cur_idx[i]];
  } 

  for(int i = 0; i < n_iterations_init; i++) {
    sample_discrete();
  }
}
    

