---
name: sindae-worked-examples
description: Annotated reference problems (four-tank index-2 DAE, Leslie-Gower ODE, fed-batch bioreactor). Use when a new problem resembles one of these and a structural template would help.
---

# SiNDAE Worked Examples

**When to use:** a new problem resembles one of these three reference problems.
They live in `sindae/example_problems.py`; read the source alongside this
summary. Each shows a different hybridization pattern to reason from by analogy.

## FourTankProblem: index-2 DAE, multi-output network

- Physics: four liquid-level states and five algebraic flow variables, with an
  equal-height constraint `x[t,0] == x[t,1]` that makes the system an index-2
  DAE.
- Unknown terms (`z_dim=2`): a pump characteristic and a discharge term, both
  learned. Illustrates a network with more than one output.
- Structural choices to copy: explicit algebraic variables (`block.u`); the
  equal-height coupling; the initial-point clamp on an algebraic variable at `t0`
  (Radau does not enforce algebraic constraints there); `Constraint.Skip` to omit
  one flow bound; `aux_vars_dim=5` with `get_aux_vars` returning the flows;
  `obs_dim=4`.
- Template for: hydraulic and network systems, any DAE with algebraic couplings,
  and problems where the unknown is more than one term.
- Typical config: `method="simultaneous"` with an exact Hessian.

## LeslieGowerProblem: single-term ODE, decomposition

- Physics: a two-state predator-prey ODE (prey and predator).
- Unknown term (`z_dim=1`): one interaction term in the predator equation, a
  modified Holling type II response. The rest of the dynamics stays mechanistic.
  A clean example of replacing the smallest uncertain term.
- Structural choices to copy: `within=pyo.NonNegativeReals` on the populations; a
  dormant Lyapunov block available behind a flag; `get_aux_vars` returning the
  Lyapunov variable.
- Template for: ecological and population models, and any case where exactly one
  closure or response function is unknown.
- Typical config: `method="decomposition"` (FERAL KKT); inspect `model.history`,
  since `termination` is `None` for decomposition.

## FedBatchBioreactorProblem: growth-rate hybridization, per-trajectory constants

- Physics: a four-state fed-batch bioreactor (biomass, product, substrate,
  volume).
- Unknown term (`z_dim=1`): the specific growth rate that multiplies biomass in
  several balances (the true relationship is Monod kinetics). This is the
  canonical serial hybrid structure: a learned kinetic rate in series with
  mechanistic material balances.
- Structural choices to copy: `domain=pyo.NonNegativeReals` on all states; a
  per-trajectory constant read from the initial conditions (the feed substrate
  `Sf = float(x0[2])`); `obs_dim=4`.
- Template for: (bio)chemical reactors and any model where an uncertain rate law
  couples into multiple balances.
- Typical config: `method="simultaneous"`. The learned rate enters
  multiplicatively here; a learned term can be additive or multiplicative
  depending on where the physics places it.

See `hybridization-strategy` for deciding which term, and how many, to hand to
the network before you code any of these.
