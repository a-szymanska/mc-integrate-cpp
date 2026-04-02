#include <vector>
double compute_corellated_error(std::vector<double>& autocov, double variance, int n_points){
    double tau_int = 1.0;

    for (int t = 1; t < autocov.size(); t++) {
        double cur_cov = autocov[t]/(n_points - t);
        if (cur_cov <= 0) { // Gets too noisy, so stop here
            break;
        }

        tau_int += 2.0 * cur_cov / variance;
    }
    return std::sqrt(variance * tau_int / n_points);
}
