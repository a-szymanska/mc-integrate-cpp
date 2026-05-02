#include <iostream>
#include <vector>
#include <cmath>

#include "../include/sample_mcmc_system.hpp"

#ifdef USE_GNUPLOT
    #include "gnuplot-iostream.h"
#endif

class IsingModel
{
public:
    IsingModel(int n_rows, int n_cols)
        : sampler(n_rows, n_cols, std::vector<int>{-1, 1},
            [this](const std::vector<std::vector<int>> &state, int x, int y) {
                return this->get_energy_contrib(state, x, y);
            })
    {}

    IsingModel(int n_rows, int n_cols, std::vector<std::vector<int>> init_state)
    : sampler(n_rows, n_cols, std::vector<int>{-1, 1},
            [this](const std::vector<std::vector<int>> &state, int x, int y) {
                return this->get_energy_contrib(state, x, y);
            },
            init_state)
    {}

    void simulate(double beta, int n_iterations, int sample_every = 100) {
        energies = {0.0};

        std::cout << "Initial state:\n";
        print_state();
    
        for (int i = 0; i < n_iterations; i++) {
            auto delta_energy = sampler();

            if (i % sample_every == 0) {
                energies.push_back(energies.back() + delta_energy);
            }
        }

        std::cout << "State after " << n_iterations << " iterations:\n";
        print_state();

        std::cout << "Total energy change: " << energies.back() << std::endl;
    }

    void plot() {
    #ifdef USE_GNUPLOT
        Gnuplot gp;
        
        int n_energies = energies.size();
        std::vector<std::pair<int, double>> data;
        data.reserve(energies.size());

        for (int i = 0; i < n_energies; i++) {
            data.emplace_back(i, energies[i]);
        }

        gp << "set title 'Energy vs Number of iterations'\n";
        gp << "set xlabel 'Number of iterations'\n";
        gp << "set ylabel 'Energy'\n";
        gp << "plot '-'\n";

        gp.send1d(data);
    #else
        std::cout << "Plotting disabled (rebuild the project with gnuplot to enable)." << std::endl;
    #endif
    }

private:
    double beta;

    std::vector<double> energies;
    McmcSystemSampler<int> sampler;

    int get_energy_contrib(const std::vector<std::vector<int>> &state, int x, int y) {
        int n_rows = state.size();
        int n_cols = state[0].size();

        int energy = 0;
        int cur_value = state[x][y];

        energy -= cur_value * state[(x - 1 + n_rows) % n_rows][y];
        energy -= cur_value * state[(x + 1) % n_rows][y];
        energy -= cur_value * state[x][(y - 1 + n_cols) % n_cols];
        energy -= cur_value * state[x][(y + 1) % n_cols];

        return energy;
    }

    void print_state() const {
        const auto & state = sampler.get_state();

        for (const auto &row : state) {
            for (int val : row) {
                std::cout << (val == 1 ? "+ " : "- ");
            }
            std::cout << "\n";
        }
        std::cout << std::endl;
    }
};


int main() {

    std::vector<std::vector<int>> init_state = {
        {1, 1, 1, 1, 1},
        {1, 1, -1, 1, 1},
        {1, -1, 1, 1, 1},
        {1, 1, -1, -1, 1},
        {1, 1, -1, 1, 1},
        {1, 1, 1, 1, -1}
    };

    IsingModel model(6, 5, init_state); // n_rows = 6, n_cols = 5
    model.simulate(0.05, 10e5);         // beta = 0.05, n_iterations = 10^5
    model.plot();

    return 0;
}