#include <iostream>
#include <vector>
#include <cmath>

#include "../include/sample_mcmc.hpp"
#include "../build/exter"

class IsingModel
{
public:
    IsingModel(int n_rows, int n_cols)
        : sampler(n_rows, n_cols, std::vector<int>{-1, 1},
            [this](const std::vector<std::vector<int>> &state, int x, int y, int new_value) {
                return this->get_energy_change(state, x, y, new_value);
            })
    {}

    IsingModel(int n_rows, int n_cols, std::vector<std::vector<int>> initial_state)
    : sampler(n_rows, n_cols, std::vector<int>{-1, 1},
            [this](const std::vector<std::vector<int>> &state, int x, int y, int new_value) {
                return this->get_energy_change(state, x, y, new_value);
            },
            initial_state)
    {}

    void simulate(double temp, int n_iterations, int sample_every = 100) {
        beta = 1.0 / std::max(temp, 1e-2);

        energies = {0.0};

        std::cout << "Initial state:\n";
        print_state();
        for (int i = 0; i < n_iterations; i++) {
            auto res = sampler();

            if (i % sample_every == 0) {
                energies.push_back(energies.back() + res.delta_energy);
            }
        }

        std::cout << "State after " << n_iterations << " iterations:\n";
        print_state();

        std::cout << "Total energy change: " << energies.back() << std::endl;
    }

    void plot() {
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
        gp << "plot '-'";

        gp.send1d(data);
    }

private:
    double beta;

    std::vector<double> energies;
    McmcSystemSampler<int> sampler;

    double get_energy_change(const std::vector<std::vector<int>> &S, int x, int y, int new_value) {
        int n_rows = S.size();
        int n_cols = S[0].size();

        int cur_value = S[x][y];
        double cur_E = 0.0, new_E = 0.0;

        if (x > 0) {
            cur_E -= cur_value * S[x - 1][y];
            new_E -= new_value * S[x - 1][y];
        }
        if (x < n_rows - 1) {
            cur_E -= cur_value * S[x + 1][y];
            new_E -= new_value * S[x + 1][y];
        }
        if (y > 0) {
            cur_E -= cur_value * S[x][y - 1];
            new_E -= new_value * S[x][y - 1];
        }
        if (y < n_cols - 1) {
            cur_E -= cur_value * S[x][y + 1];
            new_E -= new_value * S[x][y + 1];
        }
        return std::exp(-beta * (new_E - cur_E));
    }

    void print_state() const {
        const auto & S = sampler.get_state();

        for (const auto &row : S) {
            for (int val : row) {
                std::cout << (val == 1 ? "+ " : "- ");
            }
            std::cout << "\n";
        }
        std::cout << std::endl;
    }
};


int main() {

    std::vector<std::vector<int>> initial_state = {
        {1, 1, 1, 1, 1},
        {1, 1, -1, 1, 1},
        {1, -1, 1, 1, 1},
        {1, 1, -1, -1, 1},
        {1, 1, -1, 1, 1},
        {1, 1, 1, 1, -1}
    };

    IsingModel model(6, 5, initial_state);
    model.simulate(260.0, 10e5);
    model.plot();

    return 0;
}