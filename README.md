![SiNDAE](docs/images/SiNDAE_logo_dark.png)

# SiNDAE — A Simultaneous Approach for Training Neural Differential-Algebraic Equations

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)
[![arXiv](https://img.shields.io/badge/arXiv-2504.04665-b31b1b.svg)](https://arxiv.org/abs/2504.04665)
[![CI](https://github.com/llueg/SiNDAE/actions/workflows/ci.yml/badge.svg)](https://github.com/llueg/SiNDAE/actions/workflows/ci.yml)

**SiNDAE** is a Python package for hybrid modeling of dynamical systems. It learns
unknown nonlinear terms in ODE and DAE systems directly from data by embedding a
neural network inside the governing equations and training it as a single nonlinear
program (NLP). Because the mechanistic equations are kept as hard constraints, the
learned model stays physically consistent, including when predicting new operating
conditions never seen during training.

SiNDAE is the companion code to
[*A simultaneous approach for training neural differential-algebraic systems of
equations*](https://arxiv.org/abs/2504.04665) (Lueg et al., 2025).

Authors:
- [Laurens Lueg](https://github.com/llueg)
- [Nicolas Smits](https://github.com/nicksmits1)
- [Victor Alves](https://github.com/victoraalves)

## Features

- **A scikit-learn-style interface**: `HybridDAE(...)` runs the whole pipeline behind
  `fit(problem)` / `predict(new_problem)`, with every stage still configurable.
- **Two training backends** behind a symmetric API: either the *simultaneous* approach or the
  *decomposition* approach.
- **ODEs and high-index DAEs**, discretized with Pyomo collocation.
- **Bring your own data**: fit to measured time series, including the partially
  observed case where only some states are recorded.
- **Custom neural architectures** through a grey-box interface, in addition to the
  built-in `SimpleMLP`.
- **Inference under new conditions**: embed a trained model in a fresh problem and
  predict, with the mechanistic structure keeping the result physically feasible.
- **Binary-free install**: the pure-Rust [POUNCE](https://github.com/jkitchin/pounce)
  and [FERAL](https://github.com/jkitchin/feral) solvers replace HSL/MA27, so no
  licensed binaries are required.
- **Trained model distribution**: export your trained neural network as a JAX serialized .eqx   file, an ONNX file, an OMLT `NetworkDefinition`, or a JSON.

## Installation

> **Coming soon to PyPI.** Until then, use the development install from source below.

```bash
pip install sindae            # core: full POUNCE/FERAL workflow (simultaneous, decomposition, inference)
pip install "sindae[full]"    # adds mpi4py (MPI) and cyipopt (optional alternative NLP backend)
```

The core install is pure pip wheels with no system libraries or licenses, and runs the
entire pipeline (simultaneous, decomposition, grey-box, inference) on POUNCE and FERAL.
The `full` extra adds `mpi4py` (for MPI-parallel decomposition) and `cyipopt` (an optional
alternative NLP backend), whose wheels are platform-dependent; if they do not build, install
them from conda-forge and `pip install sindae` into the same environment. See
[`docs/installation.md`](docs/installation.md) for the conda route, GPU/Apple Silicon, and
troubleshooting.

For a development install from source:

```bash
git clone https://github.com/llueg/SiNDAE.git
cd SiNDAE
pip install -e ".[full,test]"
```

## Quickstart

Generate noisy data from a built-in example, fit the hybrid model, and predict under
new conditions with the `HybridDAE` wrapper:

```python
import jax
import numpy as np
import sindae as sd

jax.config.update("jax_enable_x64", True)

problem = sd.LeslieGowerProblem(nfe=40, ncp=3)      # or define your own problem (see below)
sd.generate_data(problem, noise_std=[0.05, 0.05])   # or load your own measurements

mlp = sd.SimpleMLP(in_size=2, out_size=1, widths=[16, 16],
                   activations=[jax.nn.softplus] * 2)

model = sd.HybridDAE(
    method="simultaneous",              # or "decomposition"
    net=mlp,
    train=sd.SimultaneousConfig(reg_coef=1e-3),
    smoother=sd.SmootherConfig(smooth_coef=10.0),
    pretrain=sd.PretrainConfig(epochs=200, batch_size=32, reg_coef=1e-3),
    solver_options=sd.SolverConfig(tol=1e-6, max_iter=1000, hessian_approximation='exact'),
)
model.fit(problem)                      # smoother -> pretrain -> train

new_problem = sd.LeslieGowerProblem(ics=np.array([[1.2, 0.15]]), nfe=40, ncp=3)
pred = model.predict(new_problem, slack_coef=1e-5)   # inference on new conditions
```

Change the method to `decomposition` and use `train=sd.DecompConfig(...)` to use the decomposition approach.

See the [Quickstart guide](docs/quickstart.md) for the full walkthrough.

## How it works

A typical workflow has four stages: build a problem, solve a *smoother* to get smooth
warm-start trajectories and normalization statistics, pre-train the network on those,
then train the hybrid model with one of the two methods below. `HybridDAE.fit` wraps each of these stages into one function, where the method can be specified with the flag `method=`; the entry points below give stage-level control.

| Method | Entry point |  |
|---------|-------------|------|
| Simultaneous | `HybridDAE.fit(method='simultaneous')` | Network weights, states, and algebraic variables are decision variables in a single NLP solved by POUNCE or IPOPT using either exact Hessian, or L-BFGS for the grey-box variant. |
| Decomposition | `HybridDAE.fit(method='decomposition')`| An outer Adam loop updates network weights while each inner step solves the DAE with network weights fixed and obtains gradients computing the sensitivity of the inner solve. Supports MPI across trajectories. |

Both require the network to be twice continuously differentiable. Accordingly, the activation functions available in SiNDAE consist of smooth activations (`tanh`, `softplus`, `swish`) in the `SimpleMLP` class. See
[Defining a Network Architecture](docs/api/network_architecture.md) on how to define your own network structure.

## Documentation

The complete documentation with detailed functionality explanations, examples, and optional dependencies can be found [here](https://alves-research-group.github.io/SiNDAE/).

## Examples
Rendered notebooks in [`docs/examples_gallery/`](docs/examples_gallery/) show some of the package capabilities:

| Notebook | Demonstrates |
|----------|--------------|
| `four_tank_example.ipynb` | Simultaneous training on an index-2 DAE |
| `leslie_gower_example.ipynb` | Decomposition training with a custom Lyapunov path constraint |
| `fedbatch_example.ipynb` | Fedbatch bioreactor example using measured data |
| `fedbatch_partial_obs_example.ipynb` | Fedbatch bioreactor example using only partially observed states |
| `fedbatch_validation_example.ipynb` | Fedbatch bioreactor example determining optimal network size |

The same systems are also available as runnable scripts in [`examples/`](examples/) showcasing the fully configurable workflow `HybridDAE` encapsulates:

| Script | System |
|--------|--------|
| `four_tank.py` | Four-tank hydraulic network (index-2 DAE) |
| `leslie_gower.py` | Leslie-Gower predator-prey (ODE) |
| `fedbatch.py` | Fed-batch bioreactor (ODE) |
| `example_mpi.py` | Four-tank trained over MPI ranks |

Set `METHOD = 'simul'` or `METHOD = 'decomp'` at the top of each script to switch
backends.

## Defining your own problem

Subclass `ProblemDefinition` and implement the three required methods. The network
takes `get_input_vars` as input and produces `get_output_vars`; `build_trajectory`
writes the mechanistic ODE/DAE and fixes the initial conditions.

```python
import pyomo.environ as pyo
import pyomo.dae as dae
from sindae.problem import ProblemDefinition

class MyProblem(ProblemDefinition):
    def build_trajectory(self, block, traj_idx):
        block.t    = dae.ContinuousSet(bounds=self.t_span)
        block.x    = pyo.Var(block.t, range(2), initialize=1.0)
        block.z    = pyo.Var(block.t, range(1))            # the learned term
        block.dxdt = dae.DerivativeVar(block.x, wrt=block.t)
        # ... add ODE/DAE constraints that reference block.z[t, 0] ...
        block.x[self.t_span[0], 0].fix(self.ics[traj_idx, 0])

    def get_input_vars(self, block, t):
        return [block.x[t, j] for j in range(2)]           # fed into the network

    def get_output_vars(self, block, t):
        return [block.z[t, 0]]                             # produced by the network
```

Optional overrides let you customize the observation model (`get_obs_vars`), track
extra variables (`get_aux_vars`), or define the true term for synthetic data
generation (`add_true_output_constraints`, used only by `generate_data`). See
[`sindae/example_problems.py`](sindae/example_problems.py) for complete
implementations of the four-tank DAE, Leslie-Gower ODE, and fed-batch bioreactor.

## Hybrid model development with Claude

To reduce the learning curve of the package and streamline hybridizing a model, defining a `ProblemDefinition`, selecting a solution method, and solving the model to convergence, a CLAUDE.md file along with a set of skills is included in [`sindae-skills/`](sindae-skills/). 

Copy the bundle into your own modeling project and [Claude](https://claude.com/claude-code) will ask you about the process, draft the governing equations, and, critically, render them and refine the model with you before writing or running any code.

See [`sindae-skills/README.md`](sindae-skills/README.md) for setup.

## Citation

```bibtex
@article{lueg2025simultaneous,
  title={A simultaneous approach for training neural differential-algebraic systems of equations},
  author={Lueg, Laurens R and Alves, Victor and Schicksnus, Daniel and Kitchin, John R and Laird, Carl D and Biegler, Lorenz T},
  journal={arXiv preprint arXiv:2504.04665},
  year={2025}
}
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for
details.
