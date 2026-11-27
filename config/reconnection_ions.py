# Two-species (ion) reconnection: the reconnection_viscous.py setup with the ions switched on.
#
# reconnection_viscous.py itself is unchanged. Differences from
# it: ions = 1, mass_ratio, nu_i, an ion closure, eta = 0 stated as required, and sigma, lx, ly as
# named parameters. Grid, dt formula, initial field and electron closure are the same.
#
# Cost note: 1024 x 1024 with ions is roughly twice the electron-only cost per step, and writes
# 22 fields per plotfile instead of 16. Try a smaller nx, ny first.

import numpy as np

# --- box and current sheet (lengths in electron skin depths d_e) ---------------
# At M = 25, d_i = sqrt(M) d_e = 5 d_e, so 80 x 40 d_e is 16 x 8 d_i: fine for test runs.
# Ion-scale runs want a current sheet of about 1 d_i and a box of about 10 d_i. That run is not
# set up here.
lx = 80.0      # box length in x
ly = 40.0      # box length in y; holds two current sheets, at y = ly/4 and y = 3 ly/4
sigma = 2.0    # sheet half-width: near y = ly/4, Bx ~ -B0 tanh((y - ly/4)/sigma)

nx = 1024
ny = 1024
dx = lx / nx
dy = ly / ny

# --- dt -------------------------------------------------------------------------
# PROVISIONAL: to be replaced by a two-species stability rule.
# This is still the shipped formula of reconnection_viscous.py. It has no viscous limit and no ion
# term.
dt_cfl = (dx**-2 + dy**-2) ** (-0.5)
dt = dt_cfl * 0.3 * 5
t_end = 800.0
output_every = int(t_end / dt / 200)

nu = 1.0e-2
eta = 0        # must be 0 with ions on: the two-species model is collisionless

# --- ions -----------------------------------------------------------------------
ions = 1
mass_ratio = 25.0   # M = m_i/m_e. The real ~1836 separates scales too much;
                    # 100 is typical, 25 for small test runs.
nu_i = nu           # ion viscosity, its own input; equal to nu for now

output_dir = "output_ions_M25"

B0 = 1.0
psi0 = 0.1


def _sheet_phase(y):
    return (ly / (2.0 * np.pi * sigma)) * np.cos(2.0 * np.pi * y / ly)


def Bx(x, y):
    bx_eq = B0 * np.tanh(_sheet_phase(y))
    perturb = 2.0 * (2.0 * np.pi / ly) * psi0 * np.cos(2.0 * np.pi * (x - 0.25 * lx) / lx) * np.sin(
        4.0 * np.pi * (y - 0.25 * ly) / ly
    )
    return bx_eq + perturb


def By(x, y):
    return -psi0 * (2.0 * np.pi / lx) * np.sin(2.0 * np.pi * (x - 0.25 * lx) / lx) * np.cos(
        4.0 * np.pi * (y - 0.25 * ly) / ly
    )


def Bz(x, y):
    return B0 / np.cosh(_sheet_phase(y))


# --- closures: one per species ---------------------------------------------------
# Electrons, as in reconnection_viscous.py: P_e = -nu (grad u_e + grad u_e^T). ctx["u"] is u_e.
pressure_closure_dependencies = ("u",)


def pressure_closure(ctx):
    ux = ctx["u"]["x"]
    uy = ctx["u"]["y"]
    uz = ctx["u"]["z"]
    dx = ctx["dx"]
    dy = ctx["dy"]
    p_out = ctx["p_out"]

    p_out["xx"][:] = -2.0 * nu * (ux - np.roll(ux, 1, axis=1)) / dx
    p_out["xy"][:] = -nu * ((np.roll(ux, -1, axis=0) - ux) / dy + (np.roll(uy, -1, axis=1) - uy) / dx)
    p_out["xz"][:] = -nu * (uz - np.roll(uz, 1, axis=1)) / dx
    p_out["yy"][:] = -2.0 * nu * (uy - np.roll(uy, 1, axis=0)) / dy
    p_out["yz"][:] = -nu * (uz - np.roll(uz, 1, axis=0)) / dy
    p_out["zz"][:] = 0.0

    return None


# Ions: the same body. The solver passes the ion fields under the same names (ctx["u"] is u_i,
# ctx["species"] is "ion"), so only the coefficient changes: P_i = -M nu_i (grad u_i + grad u_i^T).
# M stays separate from nu_i so either can be changed alone.
# Ion equation: dQ_i/dt = curl(u_i x Q_i) + M nu_i lap(curl u_i).
ion_pressure_closure_dependencies = ("u",)


def ion_pressure_closure(ctx):
    ux = ctx["u"]["x"]
    uy = ctx["u"]["y"]
    uz = ctx["u"]["z"]
    dx = ctx["dx"]
    dy = ctx["dy"]
    p_out = ctx["p_out"]
    visc = mass_ratio * nu_i  # ion viscous coefficient M * nu_i

    p_out["xx"][:] = -2.0 * visc * (ux - np.roll(ux, 1, axis=1)) / dx
    p_out["xy"][:] = -visc * ((np.roll(ux, -1, axis=0) - ux) / dy + (np.roll(uy, -1, axis=1) - uy) / dx)
    p_out["xz"][:] = -visc * (uz - np.roll(uz, 1, axis=1)) / dx
    p_out["yy"][:] = -2.0 * visc * (uy - np.roll(uy, 1, axis=0)) / dy
    p_out["yz"][:] = -visc * (uz - np.roll(uz, 1, axis=0)) / dy
    p_out["zz"][:] = 0.0

    return None
