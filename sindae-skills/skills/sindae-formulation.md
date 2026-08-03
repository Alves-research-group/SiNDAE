---
name: sindae-formulation
description: Translate a system of ODEs or DAEs into a SiNDAE ProblemDefinition subclass, covering the Pyomo and collocation cookbook, attaching measurements, and the self-check to run before fitting. Use whenever a problem is being defined or a formulation is suspected to be wrong.
---

# SiNDAE Formulation

**When to use:** turning approved governing equations into a `ProblemDefinition`
subclass, attaching data to it, and checking it is well posed before any fit.
Load this at workflow stage 4 (Build) and stage 5 (Validate).

Do not reach this document before the user has approved the equations. The rendering gate in `CLAUDE.md` comes first, always.

## The contract

`ProblemDefinition` (in `sindae/problem.py`) is an ABC. A subclass must
implement three methods and may override four more.

| Method | Required | Returns |
|--------|----------|---------|
| `build_trajectory(block, traj_idx)` | yes | nothing; adds vars and constraints to one trajectory block |
| `get_input_vars(block, t)` | yes | list of Pyomo Vars fed into the network at `t`, length `input_dim` |
| `get_output_vars(block, t)` | yes | list of Pyomo Vars produced by the network at `t`, length `z_dim` |
| `get_obs_vars(block, t)` | no | list of measured Vars, length `obs_dim`. Default: same as `get_input_vars` |
| `get_aux_vars(block, t)` | no | extra Vars to record in `InstanceData`, length `aux_vars_dim`. Default: `[]` |
| `add_true_output_constraints(block)` | no | pins the outputs to a known formula; used only by `generate_data` |
| `discretize(model)` | no | Lagrange-Radau collocation. Override only for a different scheme |

Everything the package does is built on these. The base class stores dimensions
and data and computes `obs_mean` / `obs_std`; it holds no physics.

## Constructor arguments

`super().__init__` takes these, in this order:

| Argument | Shape / type | Meaning |
|----------|--------------|---------|
| `ics` | `(n_traj, state_dim)` | initial conditions per trajectory. `len(ics)` sets `num_trajectories` |
| `input_dim` | int | length of `get_input_vars`, must equal the network `in_size` |
| `z_dim` | int | length of `get_output_vars`, must equal the network `out_size` |
| `t_span` | `(t0, tf)` | time horizon, shared by every trajectory |
| `nfe` | int | finite elements for collocation |
| `ncp` | int | collocation points per element |
| `obs_times` | list of `(T_i,)` arrays, one per trajectory | measurement times, may differ in length between trajectories |
| `obs_values` | list of `(T_i, obs_dim)` arrays | measurements, aligned with `obs_times` |
| `obs_dim` | int, optional | number of measured channels. Defaults to `input_dim` |
| `aux_vars_dim` | int, optional | length of `get_aux_vars` |

`obs_times` and `obs_values` may be left `None` at construction and set later,
either by hand or by `generate_data`. `HybridDAE.fit` raises if they are still
`None`.

A trajectory is one experiment, batch, or run. Every trajectory shares the same
equations, the same `t_span`, and the same discretization. What varies per
trajectory is the initial condition, and anything else you derive from
`self.ics[traj_idx]` inside `build_trajectory`.

## The skeleton

```python
import numpy as np
import pyomo.environ as pyo
import pyomo.dae as dae
from sindae.problem import ProblemDefinition

class MyProblem(ProblemDefinition):
    def __init__(self, ics, params, t_span=(0.0, 40.0), nfe=40, ncp=3,
                 obs_times=None, obs_values=None):
        super().__init__(
            np.asarray(ics),
            input_dim=3,          # states fed to the network
            z_dim=1,              # unknown terms learned
            t_span=t_span, nfe=nfe, ncp=ncp,
            obs_times=obs_times, obs_values=obs_values,
            obs_dim=3,
        )
        self.params = params

    def build_trajectory(self, block, traj_idx):
        t0 = self.t_span[0]
        x0 = self.ics[traj_idx]
        p  = self.params

        block.t    = dae.ContinuousSet(bounds=self.t_span)
        block.x    = pyo.Var(block.t, range(3), domain=pyo.NonNegativeReals,
                             initialize=1.0)
        block.z    = pyo.Var(block.t, range(1), initialize=0.1)   # the learned term
        block.dxdt = dae.DerivativeVar(block.x, wrt=block.t)

        @block.Constraint(block.t, range(3))
        def diffeq(b, t, s):
            X, S, V = b.x[t, 0], b.x[t, 1], b.x[t, 2]
            mu = b.z[t, 0]                                         # network output
            if s == 0:
                return b.dxdt[t, 0] == mu * X - p['Feed'] * X / V
            elif s == 1:
                return b.dxdt[t, 1] == p['Feed'] * (p['Sf'] - S) / V - mu * X / p['Yxs']
            else:
                return b.dxdt[t, 2] == p['Feed']

        for j in range(3):
            block.x[t0, j].fix(float(x0[j]))

    def get_input_vars(self, block, t):  return [block.x[t, j] for j in range(3)]
    def get_output_vars(self, block, t): return [block.z[t, 0]]
```

## build_trajectory rules

`build_trajectory` is called once per trajectory, **before** discretization, on
a fresh `pyo.Block`. Pyomo DAE expands every var and constraint indexed by
`block.t` onto the collocation grid afterwards.

- **The ContinuousSet must be an attribute named `t`.** `discretize` calls
  collocation with `wrt=model.trajectories[i].t`. Any other name fails.
- **Build it pre-discretization only.** Do not call the collocation transform
  yourself, and do not add anything that assumes `block.t` is already a finite
  set. The package discretizes for you.
- **Index constraints by `block.t`**, not by a list of times. That is what makes
  them expand to every collocation point.
- **Every state needs a `DerivativeVar` and a constraint that determines it.**
  A `dae.DerivativeVar` with no equation is a free variable at every point.
- **Fix the initial conditions** with `block.x[t0, j].fix(float(x0[j]))`. Cast
  to `float`; passing a numpy scalar works but a numpy array does not.
- **Per-trajectory constants come from `self.ics[traj_idx]`.** The fed-batch
  problem reads its feed concentration this way (`Sf = float(x0[2])`). This is
  the mechanism for anything that changes between experiments. There is no
  separate per-trajectory parameter argument.
- **`self.params` is yours.** Store known physical constants on the instance in
  `__init__` and read them in `build_trajectory`. Do not hard-code numbers into
  the constraint bodies.

### Reserved names

The stage builders attach their own components to each trajectory block. Do not
use these names for your own vars or constraints:

`norm_input`, `norm_output`, `norm_obs`, `nn_input_set`, `nn_output_set`,
`nn_obs_set`, `nn_z`, `z_smooth`, `dz_smooth_dt`, `z_smooth_constr`,
`norm_input_constr`, `norm_output_constr`, `norm_obs_constr`

At the model level, `traj_set`, `trajectories`, and `obj` are also taken.

## Where the network enters

The network is not written into `build_trajectory`. `build_trajectory` writes
the base DAE with the unknown term left as a **free Pyomo variable**, and the
stage builders wire the network to that variable afterwards.

So:

- `get_output_vars` returns the free variable(s) that stand in for the unknown
  term. Declare them like any other var: `block.z = pyo.Var(block.t, range(z_dim))`.
- **The output var must actually appear in a constraint.** If `block.z` is
  declared but never referenced by the dynamics, training will happily converge
  and the learned term will mean nothing. This is the single most common silent
  formulation error.
- `get_input_vars` returns what the network sees. These must be quantities that
  exist at every time point in the model. They are also normalized by the
  package using statistics from the smoother solve, so no manual scaling.
- Both are read at a specific `t` and must return the list in a stable order.
  The order defines the network input and output layout and must not change
  between calls.

### Where the learned term sits in the equation

The network output is an ordinary algebraic quantity, so it can enter however
the physics dictates:

```python
b.dxdt[t, 0] == b.z[t, 0]                      # whole right-hand side learned
b.dxdt[t, 0] == IN - OUT + b.z[t, 0]           # additive correction
b.dxdt[t, 0] == b.z[t, 0] * b.x[t, 0]          # learned rate, mechanistic structure
b.u[t, 0]    == b.z[t, 0]                      # learned algebraic closure
```

The last form, defining an algebraic variable to equal the network output and
using that variable in the dynamics, is how the four-tank problem attaches its
two learned terms. Use it when the same learned quantity appears in several
equations. Which of these to choose is a modeling decision, not a coding one:
see `sindae-hybridization-strategy`.

**Only measurable quantities may be network inputs.** A network that depends on
a state nobody can measure gives a model nobody can deploy.

## The observation model

`get_obs_vars` defines what is compared against data. The default is
`get_input_vars`, which is right only when every network input is measured.

Override it whenever the measured channels differ from the network inputs, and
pass a matching `obs_dim`:

```python
def get_input_vars(self, block, t):
    return [block.x[t, j] for j in range(4)]      # network sees all four states

def get_obs_vars(self, block, t):
    return [block.x[t, 0], block.x[t, 2]]         # only X and S are measured
# ... constructed with obs_dim=2
```

The column order of `get_obs_vars` must match the column order of
`obs_values`. There is no name-based matching. Getting this wrong fits the model
to the wrong channels and produces a plausible, wrong result.

`get_obs_vars` is not limited to states. Any Pyomo expression-valued var on the
block can be observed, which is how you handle a sensor that measures a
combination of states rather than a state directly.

### Measurement times do not need to be on the collocation grid

`_compute_norm_targets` interpolates the observations onto the model time grid
with `np.interp` per channel, so arbitrary sampling times are fine, and
trajectories may be sampled at different times and different counts.

Two consequences:

- Measurements outside `t_span` are clamped by `np.interp` rather than dropped.
  Trim your data to `t_span`, or widen `t_span`, before attaching it.
- Sparse data is silently interpolated into a dense target. Coarse sampling
  gives a smooth-looking fit that is not evidence of anything between samples.

### Partial observation changes what you must fix

With the default `unfix_io=True` the stage builders unfix every network input
and output at every time point, **including the initial conditions fixed in
`build_trajectory`**. The data anchors the trajectory instead. When some states
are unmeasured they have no such anchor, so pass `unfix_io=False` to keep the
known initial charge fixed and let the mechanistic dynamics reconstruct the
rest. See `sindae-method-selection` and `sindae-failure-playbook`.

## Auxiliary variables

`get_aux_vars` records extra quantities in `InstanceData` for plotting and
diagnostics, with `aux_vars_dim` set to their count. It has no effect on the
fit. Use it for algebraic flows, computed rates, or any derived quantity worth
inspecting after the solve. Default is `[]`, meaning nothing is recorded.

## add_true_output_constraints

Implement this only for synthetic studies. It pins the output vars to a known
formula so `generate_data` can solve the true model and produce measurements:

```python
def add_true_output_constraints(self, block):
    Ks, mu_max = self.params['Ks'], self.params['mu_max']

    @block.Constraint(block.t)
    def true_z(b, t):
        return b.z[t, 0] == mu_max * b.x[t, 2] / (Ks + b.x[t, 2])
```

It is called pre-discretization, so `block.t` is still a `ContinuousSet`. It is
never used during training. If the user has real measurements, do not write
this method, and do not invent a formula in order to have one. The base class
raises a clear `NotImplementedError` if `generate_data` is called without it.

## Attaching data

Real measurements, as lists of arrays, one entry per trajectory:

```python
import pandas as pd

raw = pd.read_csv('measurements.csv')
MEASURED = ['X', 'S']

obs_times, obs_values = [], []
for batch_id in sorted(raw['batch'].unique()):
    batch = raw[raw['batch'] == batch_id].sort_values('time')
    obs_times.append(batch['time'].to_numpy())
    obs_values.append(batch[MEASURED].to_numpy())

problem = MyProblem(ics=BATCH_ICS, params=P, obs_dim=len(MEASURED),
                    obs_times=obs_times, obs_values=obs_values)
```

`len(obs_times) == len(obs_values) == len(ics)`, and the base class asserts it.
The rows of `ics` and the entries of `obs_times` must describe the same
experiments in the same order.

Synthetic data from a known formula:

```python
true_data = sd.generate_data(problem, noise_std=[0.05, 0.05], obs_every=2)
```

This solves the true model, writes `problem.obs_times` and
`problem.obs_values` in place, and returns the noise-free trajectories.
`noise_std` must have shape `(obs_dim,)`. It returns `None` if the true-model
solve fails, which usually means the discretization is too coarse for the true
dynamics rather than that anything is wrong with the subclass.

## Collocation cookbook

`discretize` applies Lagrange-Radau collocation with `nfe` finite elements and
`ncp` points per element, giving `nfe * ncp + 1` time points per trajectory.

- **Radau does not enforce algebraic constraints at `t0`.** In a DAE with
  algebraic variables, the algebraic system at the initial point can be
  underdetermined. The four-tank problem clamps one flow explicitly to fix this:

  ```python
  @block.Constraint()
  def clamp_u1(b): return b.u[t0, 1] == b.u[t0, 0]
  ```

  Reach for this when the count in the self-check below shows one extra degree
  of freedom per trajectory in a model with algebraic variables.

- **A `DerivativeVar` at `t0` has no discretization equation.** If nothing else
  constrains it there, it is a free variable. This is harmless for a derivative
  that only feeds a diagnostic, and a genuine gap for a state derivative.

- **`nfe` and `ncp` are accuracy against cost.** The whole trajectory is
  discretized into the NLP, so doubling `nfe` roughly doubles the problem size.
  Start moderate (`nfe=20` to `40`, `ncp=3`) and raise `nfe` only if the fit
  cannot follow the data. Stiff dynamics need more elements, not more
  collocation points.

- **Discretize training and prediction independently.** A prediction problem may
  use a different `nfe` and `ncp` from the training problem. Nothing in the
  pipeline requires the grids to match, because observations are interpolated.

- **Use `Constraint.Skip`** to omit an index from an indexed constraint rule
  rather than writing a second constraint:

  ```python
  @block.Constraint(block.t, range(5))
  def flow_lb(b, t, j):
      if j == 2:
          return pyo.Constraint.Skip
      return b.u[t, j] >= 0
  ```

- **Declare physically valid domains.** `domain=pyo.NonNegativeReals` on
  concentrations, masses, and volumes keeps the interior-point solver away from
  regions where the equations are undefined. This matters more than it looks:
  the solver will find negative concentrations if you let it.

- **Guard divisions, `sqrt`, and `log`.** A term such as `Feed * P / X` is
  singular at `X = 0`. Give the denominator a strictly positive lower bound, or
  reformulate. `pyo.sqrt(b.x[t, 2])` is not differentiable at zero, so it needs
  the same treatment.

- **`initialize=` is the starting point, and it matters.** Give each var an
  order-of-magnitude-correct value. The default of zero is a bad start for
  anything that divides.

## Self-check before you fit

Run all of these. They are cheap. A fit is not.

**1. Dimensions agree.**
- `len(get_input_vars(block, t)) == input_dim == net.in_size`
- `len(get_output_vars(block, t)) == z_dim == net.out_size`
- `len(get_obs_vars(block, t)) == obs_dim == obs_values[i].shape[1]`
- `len(get_aux_vars(block, t)) == aux_vars_dim`
- `len(obs_times) == len(obs_values) == len(ics) == num_trajectories`
- `obs_times[i].shape[0] == obs_values[i].shape[0]` for every `i`

`HybridDAE.fit` checks the network match and raises. It does not check the
others.

**2. Every state derivative is constrained**, at every point where it should be.

**3. Every network output appears in a constraint.** Grep the body of
`build_trajectory` for `z[` and confirm each index is used in the dynamics.

**4. Observation columns line up.** The order of `get_obs_vars` matches the
column order of `obs_values`.

**5. Data lies inside `t_span`.** `max(obs_times[i]) <= t_span[1]` and
`min(obs_times[i]) >= t_span[0]` for every trajectory.

**6. Degrees of freedom.** Build the base model and count. With the network
outputs free and nothing else missing, the degrees of freedom should equal the
number of free network-output entries, which is
`num_trajectories * (nfe * ncp + 1) * z_dim`:

```python
import pyomo.environ as pyo

m = pyo.ConcreteModel()
m.traj_set     = pyo.RangeSet(0, problem.num_trajectories - 1)
m.trajectories = pyo.Block(m.traj_set)
for i in m.traj_set:
    problem.build_trajectory(m.trajectories[i], i)
problem.discretize(m)

nvar = sum(1 for v in m.component_data_objects(pyo.Var, active=True) if not v.fixed)
neq  = sum(1 for c in m.component_data_objects(pyo.Constraint, active=True) if c.equality)
nz   = sum(1 for i in m.traj_set
             for t in m.trajectories[i].t
             for v in problem.get_output_vars(m.trajectories[i], t) if not v.fixed)
print(f"vars {nvar}  equalities {neq}  DOF {nvar - neq}  free z {nz}")
```

Count equalities only; inequality constraints such as bounds must not enter the
balance. Interpreting the result:

- `DOF == nz`: the base DAE is square once the network is attached. This is what
  the fed-batch and four-tank problems give.
- `DOF > nz`: something is unconstrained. A small excess that scales with the
  number of trajectories is usually a derivative or algebraic variable free at
  `t0`. The Leslie-Gower problem shows exactly one extra per trajectory, from
  the dormant Lyapunov derivative `dlyap_dt[t0]`, which is benign because
  nothing depends on it. A larger excess means a missing equation.
- `DOF < nz`: over-constrained. The solve will be infeasible. Look for a
  constraint imposed at `t0` that duplicates a fixed initial condition.

**7. Then fit, and check `termination == "optimal"`.** Convergence is necessary,
not sufficient. Report per-state metrics as well, and read
`sindae-failure-playbook` on any non-optimal termination before touching the
formulation.

**8. (Optional) Bounds vs Stiffness check**: A DAE with a fast/slow timescale split (a state that
re-equilibrates in milliseconds against a horizon of minutes or hours) can report
`Converged to a point of local infeasibility` on a coarse mesh, a fine mesh, *and* a
graded mesh concentrated at the fast transitions. The interior-point path to a feasible, in-bounds solution can pass through transiently
out-of-bounds-looking intermediate iterates. Hard bounds close off that path and
surface as infeasibility, not slow convergence — indistinguishable from a genuine
resolution problem until you test for it:

1. Run check 6 (DOF) first, to rule out a structural/coding bug.
2. Re-solve the *same* mesh with the suspect bounds removed (`pyo.Var(block.t,
   initialize=...)`, no `bounds=`/`domain=`).
   - Converges cleanly, often in single-digit iterations even on a coarse mesh →
     it was the bounds. Leave them off; verify the *converged* solution's physicality
     post hoc (print each state's min/max) instead of enforcing it via bounds.
   - Still infeasible → may genuinely be a mesh/stiffness problem; move to graded
     finite elements concentrated at the fast transitions.

Do this before spending solver time on a mesh sweep. Uniform-coarse → uniform-fine →
graded does not distinguish the two causes, and can cost many minutes of solves before
showing that none of them were ever going to work when the fix was removing a
`bounds=` argument.

## Symptoms that point back here

| Symptom | Likely formulation cause |
|---------|--------------------------|
| Fit converges, learned term is meaningless | output var declared but never used in a constraint |
| `ValueError` on `fit` about `in_size` / `out_size` | network dimensions do not match `input_dim` / `z_dim` |
| Good metrics on one channel, poor on another | `get_obs_vars` order does not match `obs_values` columns |
| Infeasible immediately, before any training | over-constrained initial point, or an IC duplicated by a constraint at `t0` |
| Solve diverges with unmeasured states | `unfix_io=True` with no data anchor; see `sindae-method-selection` |
| `generate_data` returns `None` | true-model solve failed, usually too few finite elements |
| Stiff system: "local infeasibility" on every mesh tried (coarse, fine, graded) | Tight domain bounds on the fast states blocking the solver's path, not genuine stiffness — see self-check 8 |

For solver-side signatures (ASL, linear solver, iteration limits) go to
`sindae-failure-playbook` instead.
