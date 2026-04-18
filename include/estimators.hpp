/*
Estimators are given as a template argument to the integration methods.
They extract logic of calculating the errors and correlations of the samples. 
*/

#pragma once

#include <vector>
#include <cmath>

class Estimator{
  protected:
    int n_points;
    double mean = 0;
  public:
  Estimator(int new_n_points): n_points(new_n_points) {}
  virtual void add_sample(double y) = 0;
  virtual double get_variance() = 0;
  virtual double get_error() = 0;
  double get_mean(){
    return mean;
  }
};


class EstimatorNoAutocorrelations : public Estimator{
  double m2 = 0;
  int i=1;
  public:
  EstimatorNoAutocorrelations(int new_n_points)
      : Estimator(new_n_points){};

  // Welford's variance
  void add_sample(double y) override{
    double old_mean = mean;
    mean += (y - mean) / i;
    m2 += (y - old_mean) * (y-mean);
    i += 1;    
  }


  double get_variance() override{
    return m2 / (i-2);
  }

  double get_error() override{
    return std::sqrt(get_variance() / (i-1));
  }
};

/*
  Estimator simple aggregates all the samples and computes the error at the very end.
  In theory it is the most precise, but it comes with high memory cost.
*/
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


  double get_variance() override{
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

    return var * tau_int;
  }

  double get_error() override{
    return std::sqrt(get_variance() / n_points);
  }
};

//TODO: fixed batch size is temporary, we will want to estimate t-lag
constexpr int batch_size = 100;
class EstimatorNOBM : public Estimator{
  int i = 1;
  int batch_i=0;
  double batch_sum = 0.0;
  EstimatorNoAutocorrelations estimator;

  void flush() {
      if (batch_i != 0) {
          estimator.add_sample(batch_sum / batch_i);
          batch_i = 0;
          batch_sum = 0.0;
      }
  }

  public:
  EstimatorNOBM(int new_n_points) : Estimator(new_n_points), estimator(n_points / batch_size){}
 
  void add_sample(double y) override{
      mean += (y - mean) / i;
      i += 1;

      batch_sum+=y;
      batch_i+=1;

      if (batch_i == batch_size) {
          estimator.add_sample(batch_sum / batch_size);
          batch_sum = 0.0;
          batch_i = 0;
      }
  }

  double get_variance() override{
    flush();
    return estimator.get_variance();
  }

  double get_error() override{
    flush();
    return estimator.get_error();
  }
};
