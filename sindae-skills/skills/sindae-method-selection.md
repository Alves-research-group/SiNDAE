---
name: sindae-method-selection
description: Choose the SiNDAE method (simultaneous vs decomposition), solver stack, config dataclasses, unfix_io, and slack_coef. Use when configuring a HybridDAE fit or a predict call.
---

# SiNDAE Method and Solver Selection

**When to use:** configuring a `HybridDAE` fit or a `predict`. Covers `method`,
the solver stack, the config objects, `unfix_io`, and `slack_coef`.

## The wrapper call

```python
import sindae as sd

model = sd.HybridDAE(
    method="simultaneous",         # or "decomposition"
    nlp_solver="pounce",           # default; "ipopt" / "cyipopt" optional
    linear_solver="feral",         # default; "ma27" / "scipy" optional
    net=mlp,                        # a prebuilt SimpleMLP
    smoother=SmootherConfig(...),
    pretrain=PretrainConfig(...),   # pretraining always runs; epochs=0 opts out
    train=SimultaneousConfig(...),  # config type must match method
    solver_options=SolverConfig(...),
    unfix_io=True,                  # set False for partial observation
)
model.fit(problem, metrics=["rmse"])
pred = model.predict(new_problem, slack_coef=1e-5)
```

## method: simultaneous vs decomposition

- **simultaneous** solves the smoother, pretrains, then solves one large
  simultaneous NLP over the whole problem. Use it for small to moderate problems
  and when you want an exact-Hessian solve. Pass `train=SimultaneousConfig(...)`.
  `model.termination` reports the training-solve status.
    - This method should be used for small to moderate numbers of trajectories (<10)
    - More trajectories and a larger network will result in increased solution time.
    - On equally sized problems, the simultaneous approach's accuracy is higher than that
      of the decomposition approach, as solving the entire NLP typically results in better
      local solutions for appropriately sized training problems.
    - The number of interior point method iterations does not significantly increase as the
      number of NN layers increases, though the solution time does since the per iteration time
      increases. This is due to an increased cost of factorization of the linear system with 
      increasing network size.
    - Use L-BFGS hessian approximations by setting `hessian_approximation='limited-memory'`
      in `SolverConfig()` if the user requires a speed-up in solution time using the 
      simultaneous training approach.
    - Using exact hessian information by setting `hessian_approximation='exact'` will lead
      to the most accurate training results but a substantial slowdown (approx 5x-8x, case dependent) 
      compared to L-BFGS.
    - Increasing the smoothing coefficient in the `SimultaneousConfig()` tends to cause the 
      smoother pretraining and solve time to increase. Correct choice of smoother coefficient
      leads to better weight initialization since NN $(x(t), z(t))$ IO pretraining pairs are closer
      to the true fit. So inspecting solve time could be a metric to adjust smoothing coefficient. 
      This is case dependent.
    - Suggest changing to the decomposition method if the user encounters memory limitations with
      the simultaneous method. The decomposition method's solution time remains relatively constant
      with problem size and NN size due to the trivially parallelizable subproblems. Increasing problem size in
      the simultaneous approach corresponds directly to increased NLP size and difficulty.
    

- **decomposition** splits the problem into per-trajectory subproblems
  coordinated through a FERAL KKT system. Use it for many trajectories or when
  the simultaneous NLP is too large; this is the path that scales (MPI plus
  trajectory indices). Pass `train=DecompConfig(...)`. `model.termination` is
  `None` for decomposition, so inspect `model.history` instead.
    - This method should be use for large numbers of trajectories (>10)
    - This method's solution time does not change substantially with the number of
      trajectories and thus scales to larger problems.
    - As the problem size increases, so too will the number of Adam steps to reach
      the stopping criteria. If this approach is being used for a large number of 
      trajectories and is converging to poor solutions, consider increasing the maximum
      number of Adam steps by changing `DecompConfig(n_steps=)`. This will increase solution time.
    - The `DecompConfig(lr=)` and `SolverConfig(tol=, max_iter=)` should be tuned 
      as learning rate and stopping criteria would be for a traditional ML problem.
    - Parallelization using MPI is available with this method. Use this when individual
      subproblem solution time becomes large.
    - Increasing the number of layers will slow solution time more than layer width.
      Larger NNs will likely improve test accuracy at the cost of solve time.


- The `train` config type must match `method` (`SimultaneousConfig` with
  simultaneous, `DecompConfig` with decomposition), or construction raises.
- For both methods, it is reccomended to limit NN size to 6 layers at most and 
  moderate MLP width.


## Solver stack

- The defaults are pip-installable: **POUNCE** (NLP) and **FERAL** (KKT/linear).
  They run the whole pipeline under `pip install sindae`.
- `nlp_solver="ipopt"` or `"cyipopt"`, and `linear_solver="ma27"` or `"scipy"`,
  are selectable alternatives (cyipopt/IPOPT need the conda stack).
- POUNCE and IPOPT are different interior-point codes and will land on different
  local minima of a nonconvex, overparameterized fit. Compare objective values
  and KKT residuals across solvers, not the network weights.

## Grey-box vs expression-writing, and the Hessian

- The grey-box (GBM) path evaluates the network through a compiled callback. It
  needs the `pynumero_ASL` extension (see `sindae-failure-playbook`) and supplies
  no Hessian, so a limited-memory Hessian is forced. Any exact-Hessian or
  `hessian_approximation` request is inert under GBM. This is expected, not a bug.
- Exact-Hessian solves are only available on the expression-writing path (the
  network written as explicit algebraic constraints), which does not need
  `pynumero_ASL`.

## SolverConfig

`SolverConfig(tol=..., max_iter=..., mu_strategy=..., hessian_approximation=...,
print_level=..., extra_options={...})`. Pass backend-specific options through
`extra_options`. Start from defaults; tighten `tol` only when a solve is
genuinely near-degenerate, and prefer better domains and initialization first.

## unfix_io (partial observation)

- Default `True`: the network inputs and outputs are fixed to the smoothed data
  during the relevant solves.
- Set `unfix_io=False` when some states are unobserved. Unmeasured states have no
  data to anchor the fixed IO, so leaving it fixed makes the solve diverge to no
  solution on any backend. Partial observation needs `unfix_io=False` on both the
  smoother and the training solve.

## predict and slack_coef

- `predict(problem, slack_coef=..., eval_metrics=...)` runs an inference solve
  that re-simulates the trajectories with the trained network.
- `predict` uses its own `solver_options` and does not inherit the fit-time
  options.
- `slack_coef` weights how tightly the inference respects the network output. A
  tiny slack (for example `1e-5`) is near-degenerate and ill-conditioned: the
  prediction can swing with `tol`. `slack_coef=0` makes the network a hard
  constraint, which removes that sensitivity and gives a reproducible,
  solver-insensitive prediction. Prefer `slack_coef=0` when reproducibility
  matters.
