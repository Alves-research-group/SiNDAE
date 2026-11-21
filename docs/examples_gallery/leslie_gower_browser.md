# Leslie-Gower Predator-Prey in the Browser

[Click to launch](../../examples/web/leslie_gower/build/leslie-gower.html). The page runs SiNDAE's
Leslie-Gower example {cite}`bazykin1998` in the browser, with nothing to
install. It runs the `sindae` package in Pyodide, with Pyomo writing each model
and POUNCE's Python package for Pyodide solving it.

```{image} ../images/leslie_gower_browser.png
:alt: The Leslie-Gower page after its three steps, with two initial conditions
:width: 100%
```

The page follows `examples/leslie_gower.py` with `METHOD = 'simul'` and
`USE_GBM = False`. The network is written into the NLP as Pyomo expressions, and
the simultaneous approach of {cite}`lueg2025simultaneous` trains it. Each of its
three steps has its own options and runs when its button is pressed.

1. **Generate data.** `generate_data` solves the true model from each initial
   condition and adds Gaussian noise to every Nth point.
2. **Training.** `solve_smoother`, then pretraining, then
   `build_simultaneous_model` solved by POUNCE. While POUNCE solves, the page
   plots the training NLP's objective and constraint violation at each
   iteration and draws the network at that iteration. After training it shows
   each weight and bias with ±2 standard errors, and bands of ±2 standard
   deviations on the trained states and on ẑ₀.
3. **Inference.** The trained model from each initial condition beside the
   ground truth, as `make_inference_model` writes it with the network as Pyomo
   expressions in place of its grey box, with bands of ±2 standard deviations.

The standard errors come from POUNCE's sensitivity of the solved NLPs, through
pyomo-pounce. In the training NLP the measurements y are Params, so the
derivative of the trained weights with respect to them gives
Cov(θ) = (dθ/dy) Σ_y (dθ/dy)ᵀ, with Σ_y the variances of the noise step 1 adds.
In the inference the weights are Params, and each state's variance is
(dx/dθ) Cov(θ) (dx/dθ)ᵀ. The derivatives come from the KKT factorization of the
solve, so they may be inaccurate when the training uses L-BFGS in place of the
exact Hessian, the page's default.

Pyodide has no build of JAX, so the page's own Python file,
`examples/web/leslie_gower/sindae_page.py`, builds the network and pretrains it
in NumPy, and passes SiNDAE's functions an `NLPSolver` that solves each model
with POUNCE in Pyodide. The tables report the RMSE and R² of all trajectories
together. Step 3's table also reports the share of the ground truth's points
inside the bands. SiNDAE's own metrics give the MSE, RMSE and MAE of each trajectory.

## Building the page

```bash
node examples/web/leslie_gower/build.mjs
```

`build.mjs` downloads Pyodide 0.28.3, its NumPy wheel and the Pyomo 6.10.1
wheel, checking each against its SHA-256. It writes everything the page runs
into one HTML file, `examples/web/leslie_gower/build/leslie-gower.html`, which
opens from disk. That includes this repository's `sindae` package and POUNCE's
two wheels for Pyodide from `examples/web/leslie_gower/pounce/`. The docs
workflow builds it before the site, and the site build copies it into the site
as the file this page links to.

`pounce/` holds `pounce_solver`, POUNCE's Python package compiled for Pyodide
0.28.3, and `pyomo_pounce`, its Pyomo interface. To move to a new POUNCE
release, copy its Pyodide wheel from `crates/pounce-wasm/web-python/wheels/` in
POUNCE's repository, build its pyomo-pounce wheel with
`pip wheel --no-deps pyomo-pounce/`, put both in `pounce/`, and change their
names in `build.mjs` and `worker.js`.
