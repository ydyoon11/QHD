#pragma once

#include <AMReX_Array.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_FFT.H>
#include <AMReX_Geometry.H>
#include <AMReX_IntVect.H>
#include <AMReX_MultiFab.H>

#include <memory>
#include <string>

#include "Config.hpp"
#include "VectorOps.hpp"

class PressureClosureEngine;

// Which fluid a pressure closure is for. Each species has its own closure.
enum class Species { electron = 0, ion = 1 };

class Simulation {
public:
    explicit Simulation(const SimulationConfig& cfg);
    ~Simulation();

    void run();

private:
    using VectorField = vecops::VectorField2D;
    using PressureTensor = vecops::SymmetricTensorField2D;

    // q, u, p are the electron fields (Q_e, u_e, P_e). The *_i fields are the ions,
    // allocated only when cfg_.ions = 1.
    struct StateData {
        VectorField b;
        VectorField q;
        VectorField u;
        PressureTensor p;
        VectorField q_i;
        VectorField u_i;
        PressureTensor p_i;
    };

    struct StageData {
        VectorField rhs;
        VectorField rk_q0;
        VectorField rk_accum;
        VectorField rhs_i;
        VectorField rk_q0_i;
        VectorField rk_accum_i;
        VectorField work_b;
        VectorField work_q;
        VectorField work_cross;
        VectorField work_divp;
    };

    struct LevelData {
        amrex::Geometry geom;
        amrex::BoxArray ba;
        amrex::DistributionMapping dmap;

        StateData state;
        StageData stage;
    };

    struct GatheredPressureState {
        amrex::BoxArray ba;
        amrex::DistributionMapping dmap;
        bool defined = false;
        amrex::Array<amrex::MultiFab, 3> b;
        amrex::Array<amrex::MultiFab, 3> q;
        amrex::Array<amrex::MultiFab, 3> u;
        amrex::Array<amrex::MultiFab, 6> p_in;
        amrex::Array<amrex::MultiFab, 6> p_out;
    };

    void define_level(const amrex::Geometry& geom, const amrex::BoxArray& ba);
    void define_pressure_gather_buffers(GatheredPressureState& gather);
    void print_closure_choice() const;
    void initialize_state();
    void initialize_analytic_magnetic_field();
    void initialize_state_from_python_namelist();
    void load_array_into_multifab(const std::string& path,
                                  amrex::MultiFab& mf,
                                  int expected_ny,
                                  int expected_nx,
                                  int component = 0,
                                  bool include_ghosts = false) const;

    void fill_level_ghosts(VectorField& field);
    void fill_level_ghosts(PressureTensor& field);
    void compute_u_from_b(const VectorField& b, VectorField& u, StageData& stage);
    void compute_q_from_b(const VectorField& b, VectorField& u, VectorField& q, StageData& stage);
    void compute_default_pressure_from_filled(const VectorField& u, PressureTensor& p, amrex::Real nu);
    void update_pressure(StateData& state, amrex::Real time, amrex::Real dt, int step);
    void update_ion_pressure(StateData& state, amrex::Real time, amrex::Real dt, int step);
    void evaluate_python_closure(Species species,
                                 const VectorField& b,
                                 const VectorField& q,
                                 const VectorField& u,
                                 PressureTensor& p,
                                 GatheredPressureState& gather,
                                 amrex::Real time,
                                 amrex::Real dt,
                                 int step);
    void compute_div_pressure_from_filled(const PressureTensor& p, VectorField& divp);
    void compute_rhs(const StateData& state, StageData& stage);
    void compute_species_rhs(const VectorField& u,
                             const VectorField& q,
                             const PressureTensor& p,
                             const VectorField& b,
                             amrex::Real eta,
                             VectorField& rhs,
                             StageData& stage);
    void refresh_derived_state(StateData& state,
                               StageData& stage,
                               amrex::Real time,
                               amrex::Real dt,
                               int step);
    void refresh_derived_state_timed(StateData& state,
                                     StageData& stage,
                                     amrex::Real time,
                                     amrex::Real dt,
                                     int step,
                                     amrex::Real& helmholtz_wall_total,
                                     amrex::Real& velocity_wall_total,
                                     amrex::Real& pressure_wall_total);
    void evaluate_rhs_stage(StateData& state,
                            StageData& stage,
                            amrex::Real time,
                            amrex::Real dt,
                            int step,
                            amrex::Real& rhs_wall_total,
                            amrex::Real& helmholtz_wall_total,
                            amrex::Real& velocity_wall_total,
                            amrex::Real& pressure_wall_total);
    void advance_rk4(amrex::Real time,
                     amrex::Real dt,
                     int step,
                     amrex::Real& rhs_wall_total,
                     amrex::Real& helmholtz_wall_total,
                     amrex::Real& velocity_wall_total,
                     amrex::Real& pressure_wall_total);

    void solve_helmholtz(const VectorField& q, VectorField& b);
    void solve_helmholtz_ions(const VectorField& q_e,
                              const VectorField& q_i,
                              VectorField& b,
                              StageData& stage);
    void compute_ion_velocity(StateData& state, StageData& stage);

    void write_output(int step, amrex::Real time);
    void print_ion_diagnostics(int step, amrex::Real time);
    amrex::Real advective_number(const VectorField& u, amrex::Real dt) const;
    void print_dt_startup() const;
    void print_dt_diagnostics(int step, amrex::Real time) const;

    SimulationConfig cfg_;
    amrex::IntVect ng_{AMREX_D_DECL(2, 2, 0)};
    LevelData level_;
    GatheredPressureState pressure_gather_;    // electron closure's gathered copies (B, Q_e, u_e, P_e)
    GatheredPressureState pressure_gather_i_;  // ion closure's gathered copies (B, Q_i, u_i, P_i)
    std::unique_ptr<amrex::FFT::R2C<amrex::Real>> helmholtz_fft_;
    std::unique_ptr<PressureClosureEngine> pressure_engine_;
    bool init_has_b_ = false;
    bool init_has_q_ = false;
    bool init_b_ghosts_ready_ = false;
    bool has_python_pressure_closure_ = false;  // electrons: config defines pressure_closure
    bool has_python_ion_closure_ = false;       // ions: config defines ion_pressure_closure (ions = 1 only)
};
