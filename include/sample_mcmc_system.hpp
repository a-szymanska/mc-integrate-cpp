/*
The sampling algorithms for importance Monte Carlo methods.
*/

#pragma once

#include <random>
#include <vector>
#include <functional>
#include <optional>
#include <type_traits>

constexpr static int kNumIterationsInit = 1000;

// --------------- Metropolis-Hastings in 2D (MCMC) ---------------
/*
McmcSystemSampler implements a Metropolis–Hastings sampler for a system,
where each entry of a matrix takes values from a finite set and the probability (energy)
of the system depends on the interactions between the entries.

It is suitable for simulating systems like Ising models - see: examples/ising.cpp.
*/

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
    /*
    Initializes the system randomly and performs an initial number of iterations to move
    toward a stationary distribution. Useful for sampling or when no prior state is known.
    */
    McmcSystemSampler(
        int n_rows, int n_cols, const std::vector<T> &values,
        std::function<double(const std::vector<std::vector<T>> &, int, int, T)>);

    /*
    Starts from the provided initial state with no burn-in. Useful for radom simulation of a system.
    */
    McmcSystemSampler(
        int n_rows, int n_cols, const std::vector<T> &values,
        std::function<double(const std::vector<std::vector<T>> &, int, int, T)>,
        const std::vector<std::vector<T>> &initial_state);

    void set_beta(double beta) {
        this->beta = beta;
    }

    std::vector<std::vector<T>> get_state() const {
        return state;
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

    /*
    The function to compute the relative probability of changing a single entry.
    Args: current state, coordinates of the entry to change, new value for the entry
    */
    std::function<double(const std::vector<std::vector<T>> &, int, int, T)> get_energy_change;

    const std::vector<T> values;
    std::vector<std::vector<int>> state; // State of the system
    double beta = 0.05; // Inverse temperature

    McmcSystemChange<T> sample();
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
    state.resize(n_rows);
    for (int i = 0; i < n_rows; i++) {
        state[i].resize(n_cols);
        for (int j = 0; j < n_cols; j++) {
            state[i][j] = values[dist_idx(mt)];
        }
    }

    for (int i = 0; i < kNumIterationsInit; i++) {
        sample();
    }
}

template <typename T>
McmcSystemSampler<T>::McmcSystemSampler(int n_rows, int n_cols, const std::vector<T> &values, std::function<double(const std::vector<std::vector<T>> &, int, int, T)> get_energy_change, const std::vector<std::vector<T>> &initial_state)
    : values(values),
      get_energy_change(get_energy_change),
      dist_idx(0, static_cast<int>(values.size()) - 1),
      dist_prob(0.0, 1.0),
      dist_x(0, n_rows - 1),
      dist_y(0, n_cols - 1)
{
    if (initial_state.size() != static_cast<size_t>(n_rows) || initial_state[0].size() != static_cast<size_t>(n_cols)) {
        throw std::runtime_error("Initial state dimensions do not match (n_rows, n_cols)");
    }

    this->state = initial_state;
}

template <typename T>
McmcSystemChange<T> McmcSystemSampler<T>::sample()
{
    int x = dist_x(mt);
    int y = dist_y(mt);
    int next_idx = dist_idx(mt);

    double delta_energy = get_energy_change(state, x, y, values[next_idx]);
    double p_accept = std::min(1.0, std::exp(-beta * delta_energy));
    if (dist_prob(mt) <= p_accept) {
        state[x][y] = values[next_idx];
    } else {
        delta_energy = 0.0;
    }

    return {delta_energy, x, y, state[x][y]};
}
