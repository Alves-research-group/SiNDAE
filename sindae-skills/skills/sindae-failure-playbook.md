---
name: sindae-failure-playbook
description: Map SiNDAE solver terminations and error signatures to fixes. Use whenever a solve does not reach optimal, crashes, or diverges.
---

# SiNDAE Failure Playbook

**When to use:** a solve does not reach `optimal`, crashes, or diverges. Match
the signature to the cause and fix before changing the formulation.

## "Cannot load the PyNumero ASL interface"

- Cause: the grey-box, decomposition, or `return_nlp` path builds an in-memory
  PyomoNLP that needs the compiled `pynumero_ASL` extension, which ships in no
  pip or conda pyomo package.
- Fix: build it once with `pyomo build-extensions` (needs a C compiler; writes
  `~/.pyomo/lib/libpynumero_ASL*`). Fresh pip installs and CI do not have it.
- Note: the expression-writing POUNCE path (`use_gbm=False`) writes a plain `.nl`
  file and does not need ASL. If you only need that path you can avoid the build.

## Partial-observation solve returns no solution or diverges on every backend

- Cause: `unfix_io=True` (the default) fixes the network IO to data, but
  unobserved states have no data anchor.
- Fix: set `unfix_io=False` on both the smoother and the training solve. See
  `sindae-method-selection`.

## An exact-Hessian request has no effect

- Cause: the solve is on the grey-box path, which supplies no Hessian, so
  limited-memory is forced.
- Fix: none needed; this is expected. Use the expression-writing path if you
  need an exact Hessian.

## Training solve does not reach `optimal` (simultaneous)

- First check the formulation basics: initial conditions fixed, physical domains
  set, sensible `initialize=` on states and on `z`, and enough finite elements.
  See `sindae-formulation`.
- If the solve reaches a valid KKT point at a different objective than a
  reference solver, that is often fine. The fit is a nonconvex, overparameterized
  NLP with many minima. Compare the objective value and KKT residual, not the
  weights. POUNCE and IPOPT routinely disagree on the variables while agreeing on
  the objective.

## Singular KKT matrix or linear-solver return code 3 (decomposition)

- Cause: the decomposition KKT sparsity pattern changes across training steps
  (the Hessian and the bound-barrier diagonal gain and lose nonzeros). A linear
  solver that caches a symbolic analysis on the first pattern (MA27) goes
  singular on later steps.
- Fix: use the default FERAL (or scipy), which re-derive the symbolic
  factorization internally each step. Do not add a "symbolic once" optimization.

## generate_data or a true-model solve is infeasible at low resolution

- Cause: too few finite elements to represent the true dynamics.
- Fix: raise `nfe`. For reference, Leslie-Gower needs `nfe=15` rather than `10`
  for `generate_data`.

## fit(metrics=...) raises a shape mismatch

- Cause: an older version matched observation times to the training collocation
  grid by exact float membership, which fails whenever training is
  re-discretized onto a different grid.
- Fix: update to a SiNDAE version that interpolates the prediction onto the
  observation times. If you see this on current code, report it.

## predict output changes when you change `tol`

- Cause: a tiny `slack_coef` inference is ill-conditioned.
- Fix: use `slack_coef=0` (hard network constraint) for a solver-insensitive
  prediction, or match the `solver_options` you used at fit time. See
  `sindae-method-selection`.

## Export to OMLT or ONNX rejects an activation

- Cause: OMLT has smooth formulations only for a limited set of activations;
  `swish` is not supported, and the round-trip fully supports only `softplus` and
  `tanh`.
- Fix: build the network with `softplus` or `tanh` if you plan to export to OMLT.

## Two habits that prevent false conclusions

- Re-execute notebooks before trusting stored output. Stored cell output can
  reflect old code and lie about the current result.
- A converged solve is not a correct model. Always check `termination`, per-state
  metrics against held-out data, and residual sanity before reporting success.
