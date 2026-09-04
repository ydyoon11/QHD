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

class Simulation {
public:
    explicit Simulation(const SimulationConfig& cfg);
    ~Simulation();

    void run();

private:
    using VectorField = vecops::VectorField2D;
    using PressureTensor = vecops::SymmetricTensorField2D;

    struct StateData {
        VectorField b;
        VectorField q;
        VectorField u;
        PressureTensor p;
    };

    struct StageData {
        VectorField rhs;
        VectorField rk_q0;
        VectorField rk_accum;
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
    void define_pressure_gather_buffers();
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
    void compute_default_pressure_from_filled(const VectorField& u, PressureTensor& p);
    void update_pressure(StateData& state, amrex::Real time, amrex::Real dt, int step);
    void compute_div_pressure_from_filled(const PressureTensor& p, VectorField& divp);
    void compute_rhs(const StateData& state, VectorField& rhs, StageData& stage);
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

    void write_output(int step, amrex::Real time);

    SimulationConfig cfg_;
    amrex::IntVect ng_{AMREX_D_DECL(2, 2, 0)};
    LevelData level_;
    GatheredPressureState pressure_gather_;
    std::unique_ptr<amrex::FFT::R2C<amrex::Real>> helmholtz_fft_;
    std::unique_ptr<PressureClosureEngine> pressure_engine_;
    bool init_has_b_ = false;
    bool init_has_q_ = false;
    bool init_b_ghosts_ready_ = false;
    bool has_python_pressure_closure_ = false;
};
