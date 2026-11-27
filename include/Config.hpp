#pragma once

#include <string>

struct SimulationConfig {
    std::string config_path;
    std::string config_dir;

    int nx = 256;
    int ny = 256;
    double lx = 20.0;
    double ly = 20.0;

    double dt = 0.01;
    double t_end = 1.0;
    int output_every = 25;

    double nu = 1.0e-3;
    double eta = 1.0e-3;

    // Two-species (ion) model. ions = 0 runs the electron-only solver unchanged.
    int ions = 0;
    double mass_ratio = 0.0;  // M = m_i/m_e; must be set > 0 when ions = 1
    double nu_i = 0.0;        // ion viscosity; set to nu when the config omits it

    double init_b0 = 1.0;
    double init_sigma = 2.0;
    double init_perturbation = 0.05;
    std::string init_python_namelist;

    std::string output_dir = "output";

    static SimulationConfig from_file(const std::string& path);
};
