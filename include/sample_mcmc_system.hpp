/*
McmcSystemSampler implements a Metropolis–Hastings sampler for a system,
where each entry of a matrix takes values from a finite set and the probability (energy)
of the system depends on the interactions between the entries.

It is suitable for simulating systems like Ising models - see: examples/ising.cpp.
*/

#pragma once

#include <random>
#include <vector>
#include <queue>
#include <array>
#include <unordered_set>
#include <functional>
#include <optional>
#include <type_traits>

constexpr static int kNumIterationsInit = 1000;

struct PairHash
{
    std::size_t operator()(const std::pair<int, int> &p) const {
        return std::hash<int>()(p.first) ^ (std::hash<int>()(p.second) << 1);
    }
};

// --------------- Metropolis-Hastings in 2D (MCMC) ---------------

template <typename T = double>
class McmcSystemSampler
{
public:
    /*
    Initializes the system randomly and performs an initial number of iterations to move
    toward a stationary distribution. Useful for sampling or when no prior state is known.
    */
    McmcSystemSampler(
        size_t n_rows, size_t n_cols, const std::vector<T> &values,
        std::function<double(const std::vector<std::vector<T>> &, int, int)>);

    /*
    Starts from the provided initial state with no burn-in. Useful for random simulation of a system.
    */
    McmcSystemSampler(
        size_t n_rows, size_t n_cols, const std::vector<T> &values,
        std::function<double(const std::vector<std::vector<T>> &, int, int)>,
        const std::vector<std::vector<T>> &initial_state);

    void set_beta(double beta) {
        this->beta = beta;
    }

    std::vector<std::vector<T>> get_state() const {
        return state;
    }

    double operator()() {
        return sample();
    }

private:
    static std::mt19937 mt;
    std::uniform_real_distribution<double> dist_prob;
    std::uniform_int_distribution<int> dist_x;
    std::uniform_int_distribution<int> dist_y;
    std::uniform_int_distribution<int> dist_idx;

    /*
    The function to compute the energy contribution of a single entry.
    Args: current state, coordinates of the entry
    */
    std::function<double(const std::vector<std::vector<T>> &, int, int)> get_energy_contrib;

    const std::vector<T> values;
    std::vector<std::vector<int>> state; // State of the system
    size_t n_rows, n_cols;
    double beta = 0.05;

    double sample();
};

template <typename T>
std::mt19937 McmcSystemSampler<T>::mt{std::random_device{}()};

template <typename T>
McmcSystemSampler<T>::McmcSystemSampler(size_t n_rows, size_t n_cols, const std::vector<T> &values, std::function<double(const std::vector<std::vector<T>> &, int, int)> get_energy_contrib)
    : n_rows(n_rows), n_cols(n_cols),
      values(values),
      get_energy_contrib(get_energy_contrib),
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
McmcSystemSampler<T>::McmcSystemSampler(size_t n_rows, size_t n_cols, const std::vector<T> &values, std::function<double(const std::vector<std::vector<T>> &, int, int)> get_energy_contrib, const std::vector<std::vector<T>> &initial_state)
    : n_rows(n_rows), n_cols(n_cols),
      values(values),
      get_energy_contrib(get_energy_contrib),
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
double McmcSystemSampler<T>::sample()
{
    auto get_neighbors = [&](int i, int j)
    {
        return std::array<std::pair<int, int>, 4>{
            std::make_pair((i + 1) % n_rows, j),
            std::make_pair((i - 1 + n_rows) % n_rows, j),
            std::make_pair(i, (j + 1) % n_cols),
            std::make_pair(i, (j - 1 + n_cols) % n_cols)};
    };

    const double p_add = 1.0 - std::exp(-2.0 * beta);
    
    int x1 = dist_x(mt);
    int x2 = dist_y(mt);
    T seed_value = state[x1][x2];

    std::queue<std::pair<int, int>> queue({{x1, x2}});
    std::unordered_set<std::pair<int, int>, PairHash> cluster({{x1, x2}});

    while (!queue.empty()) {
        auto [y1, y2] = queue.front();
        queue.pop();

        for (auto [z1, z2] : get_neighbors(y1, y2)) {
            if (cluster.find({z1, z2}) == cluster.end() && state[z1][z2] == seed_value) {
                if (dist_prob(mt) < p_add) {
                    cluster.insert({z1, z2});
                    queue.emplace(z1, z2);
                }
            }
        }
    }

    // Get energy contribution of the cluster before the change
    double delta_energy = 0.0;
    for (auto [y1, y2] : cluster) {
        delta_energy -= get_energy_contrib(state, y1, y2);
    }

    // Chose a random uniform value different from the seed value
    int new_idx = dist_idx(mt) % (static_cast<int>(values.size()) - 1);
    if (values[new_idx] == seed_value) {
        ++new_idx;
    }
    T new_value = values[new_idx];
    for (auto [y1, y2] : cluster) {
        state[y1][y2] = new_value;
    }

    // Get contribution of the cluster after the change
    for (auto [y1, j2] : cluster) {
        delta_energy += get_energy_contrib(state, y1, j2);
    }

    return delta_energy;
}
