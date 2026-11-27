# Two-species (ion) QHD: change record

Adds a mobile ion fluid, switched on with `ions = 1`. With `ions = 0` (the default) the
electron-only path is unchanged.

## Equations

Normalised to d_e and ω_ce; M = m_i/m_e.

```
1. state                     Q_e = ω_e − B,   Q_i = M ω_i + B,   ∇×B = u_i − u_e
2. B:    ∇²B − (1 + 1/M) B = Q_e − Q_i/M
3. u_i:  ω_i = (Q_i − B)/M,   ∇²u_i = −∇×ω_i   (∇·u_i = 0);    u_e = u_i − ∇×B
4.       ∂Q_e/∂t = ∇×(u_e × Q_e) + ν_e ∇²(∇×u_e)
         ∂Q_i/∂t = ∇×(u_i × Q_i) + M ν_i ∇²(∇×u_i)
```

## Config keys

- `ions` (0/1, default 0), `mass_ratio` (M > 0), `nu_i` (default `nu`).
- With ions on, `eta` must be set to 0 (its default is 1e-3), and the initial condition must give B.

## Solver

- Steps 2 and 3 are new solves (`solve_helmholtz_ions`, `compute_ion_velocity`). The k = 0 ion
  velocity is set to zero (plasma at rest on average).
- Step 4: the RHS is computed once per species (`compute_species_rhs`), and RK4 advances Q_i
  beside Q_e.
- At t = 0, Q_i = B and u_i = 0 (ions at rest).

## Pressure closures

- A config may define `ion_pressure_closure` beside `pressure_closure`. Its ctx has the same keys,
  filled with ion fields (`q` = Q_i, `u` = u_i, `species` = `"ion"`), so one body can serve both.
- Without one, the ions use the built-in viscous stencil with coefficient M·ν_i.
- M is kept separate from ν_i, so the ion mass can be changed with everything else fixed.

## Example config

`config/reconnection_ions.py`: the `reconnection_viscous.py` setup with ions on (M = 25, ν_i = ν,
η = 0), and `sigma`, `lx`, `ly` as named parameters. Its dt is still the shipped formula; a
two-species stability rule will come in a separate change.

## Output and diagnostics

Ion fields `Qix Qiy Qiz Uix Uiy Uiz` in the plotfiles; `[ions]` residual checks and `[dt]`
stability numbers in the log.
