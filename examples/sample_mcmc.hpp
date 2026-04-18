/*
The sampling algorithms for importance Monte Carlo methods.
*/

#pragma once

#include <random>
#include <vector>
#include <functional>
#include <optional>
#include <type_traits>

// --------- Metropolis-Hastings in 2D (MCMC) --------

template <typename T>
struct McmcSystemChange
{
    double delta_energy;
    int x;
    int y;
    T value;
};

template <typename T = double>
class McmcSystemSampler
{
public:
    McmcSystemSampler(int n_rows, int n_cols, const std::vector<T> &values, std::function<double(const std::vector<std::vector<T>> &, int, int, T)>);
    McmcSystemSampler(int n_rows, int n_cols, const std::vector<T> &values, std::function<double(const std::vector<std::vector<T>> &, int, int, T)>, const std::vector<std::vector<T>> &S);

    std::vector<std::vector<T>> get_state() const {
        return S;
    }

    McmcSystemChange<T> operator()() {
        return sample();
    }

private:
    static std::mt19937 mt;
    std::uniform_real_distribution<double> dist_prob;
    std::uniform_int_distribution<int> dist_x;
    std::uniform_int_distribution<int> dist_y;
    std::uniform_int_distribution<int> dist_idx;

    constexpr static int kNumIterationsInit = 1000;

    McmcSystemChange<T> sample();

    std::function<double(const std::vector<std::vector<T>> &, int, int, T)> get_energy_change;
    // const std::vector<std::vector<T>> &S, int x, int y, T new value
    
    const std::vector<T> values;
    std::vector<std::vector<int>> S; // State of the system
};

template <typename T>
std::mt19937 McmcSystemSampler<T>::mt{std::random_device{}()};

template <typename T>
McmcSystemSampler<T>::McmcSystemSampler(int n_rows, int n_cols, const std::vector<T> &values, std::function<double(const std::vector<std::vector<T>> &, int, int, T)> get_energy_change)
    : values(values),
      get_energy_change(get_energy_change),
      dist_idx(0, static_cast<int>(values.size()) - 1),
      dist_prob(0.0, 1.0),
      dist_x(0, n_rows - 1),
      dist_y(0, n_cols - 1)
{
    // Initialise the system randomly
    S.resize(n_rows);
    for (int i = 0; i < n_rows; i++) {
        S[i].resize(n_cols);
        for (int j = 0; j < n_cols; j++) {
            S[i][j] = values[dist_idx(mt)];
        }
    }

    for (int i = 0; i < kNumIterationsInit; i++) {
        sample();
    }
}

template <typename T>
McmcSystemSampler<T>::McmcSystemSampler(int n_rows, int n_cols, const std::vector<T> &values, std::function<double(const std::vector<std::vector<T>> &, int, int, T)> get_energy_change, const std::vector<std::vector<T>> &S)
    : values(values),
      get_energy_change(get_energy_change),
      dist_idx(0, static_cast<int>(values.size()) - 1),
      dist_prob(0.0, 1.0),
      dist_x(0, n_rows - 1),
      dist_y(0, n_cols - 1)
{
    if (S.size() != static_cast<size_t>(n_rows) || S[0].size() != static_cast<size_t>(n_cols)) {
        throw std::runtime_error("Initial state dimensions do not match (n_rows, n_cols)");
    }

    this->S = S;
}

template <typename T>
McmcSystemChange<T> McmcSystemSampler<T>::sample()
{
    int x = dist_x(mt);
    int y = dist_y(mt);
    int next_idx = dist_idx(mt);

    double delta_energy = get_energy_change(S, x, y, values[next_idx]);
    double p_accept = std::min(1.0, delta_energy);
    if (dist_prob(mt) <= p_accept) {
        S[x][y] = values[next_idx];
    }

    return {delta_energy, x, y, S[x][y]};
}
