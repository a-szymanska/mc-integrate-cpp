#pragma once

#include <vector>

class Estimator{
  protected:
    int n_points;
  public:
  double mean = 0;
  Estimator(int new_n_points): n_points(new_n_points) {}
  virtual void add_sample(double y) = 0;
  virtual double get_error() const = 0;
};


class EstimatorNoAutocorrelations : public Estimator{
  double m2 = 0;
  int i=1;
  public:

  // Welford's variance
  void add_sample(double y) override{
    double old_mean = mean;
    mean += (y - mean) / i;
    m2 += (y - old_mean) * (y-mean);
    i += 1;    
  }


  double get_error() const override{
    return m2 / (n_points-1);
  }
};


class EstimatorSimple : public Estimator{
  double m2 = 0;
  int i=1;
  std::vector<double> f_values;

  public:
  EstimatorSimple(int new_n_points)
      : Estimator(new_n_points){
      f_values.reserve(n_points);
  }


  // Welford's variance
  void add_sample(double y) override{
    double old_mean = mean;
    mean += (y - mean) / i;
    m2 += (y - old_mean) * (y-mean);
    f_values.push_back(y);
    i += 1;    
  }


  double get_error() const override{
    double var = m2 / (n_points - 1);

    // Computing autocorrelation time
    int max_lag = std::min(1000, n_points / 2);
    double tau_int = 1.0;

    for (int t = 1; t < max_lag; t++) {
        double autocov = 0.0;
        for (int j = 0; j < n_points - t; j++) {
            autocov += (f_values[j] - mean) * (f_values[j + t] - mean);
        }
        autocov /= (n_points - t);
        if (autocov <= 0) { // Gets too noisy, so stop here
            break;
        }

        tau_int += 2.0 * autocov / var;
    }

    return std::sqrt(var * tau_int / n_points);
  }
};
