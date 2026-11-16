"""
train.py  (sequential approach)

Standard sequential neural-DAE training baseline: the classic neural-ODE
recipe.  The ODE is integrated forward with the neural network embedded in the
right-hand side; the data-fit loss (MSE on observed states) is backpropagated
through the integrator (diffrax adjoint) and the NN parameters are updated with
Adam (optax).

This is the appropriate baseline for the bi-level decomposition: both perform
gradient descent on the NN parameters in parameter-space.  The difference is
what provides the gradient — here, an explicit forward integration; in the
decomposition, the solution of a constrained inner NLP.  The sequential
approach is fast and GPU-friendly but cannot enforce algebraic / inequality
constraints (only soft penalties) and cannot integrate higher-index DAEs
without manual index reduction (see ``ProblemDefinition.rhs_jax``).

API
---
  SequentialConfig          — hyperparameters for the sequential training loop
  train_sequential          — main training function -> (mlp, history)
  integrate_to_instance_data — forward-integrate a trained NN -> InstanceData

Notes
-----
* The NN operates in normalised space, matching the simultaneous / decomposition
  models and ``pretrain_mlp``:  norm_x = (x - in_mean)/in_std;
  z = out_mean + out_std * mlp(norm_x).  ``data`` supplies the statistics so the
  comparison across methods is apples-to-apples.
* With ``penalty_coef = 0`` this is the *pure* standard sequential baseline.
  With ``penalty_coef > 0`` constraint violations are softly penalised — the
  "penalty-based sequential" variant.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import diffrax

from sindae.data_utils import InstanceData, TrajectoryData
from sindae.nn_utils import SimpleMLP
from sindae.problem import ProblemDefinition

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

@dataclass
class SequentialConfig:
    """Hyperparameters for the sequential (forward-integration) training loop."""
    n_steps:        int   = 2000
    lr:             float = 1e-3
    grad_clip_norm: float = np.inf     # set to np.inf to disable clipping
    param_reg_coef: float = 0.0        # L2 regularisation on NN params

    # Soft constraint penalty.  0.0 => pure standard sequential baseline.
    # penalty term = penalty_coef * mean(relu(-g)^2) over the observation grid,
    # where g = problem.path_constraint_residuals_jax(...) must be >= 0.
    penalty_coef:   float = 0.0

    # Curriculum (Kidger neural-ODE recipe): train on a leading fraction of each
    # trajectory first, then the full trajectory.  warmup_steps=0 disables it.
    warmup_frac:    float = 0.3
    warmup_steps:   int   = 0

    # diffrax integrator configuration
    solver:         str   = 'tsit5'    # 'tsit5' | 'dopri5' | 'kvaerno5' (stiff)
    rtol:           float = 1e-6
    atol:           float = 1e-8
    max_steps:      int   = 16384
    adjoint:        str   = 'recursive_checkpoint'  # | 'backsolve' | 'direct'

    # diagnostics / stopping
    eval_every:     int   = 25         # how often to record dense constraint diagnostics
    n_dense:        int   = 100        # dense grid points for constraint tracking
    patience:       int   = 0          # early stop on data-fit stagnation (0 = off)
    log_every:      int   = 50

    # Optional learning-rate schedule (callable step -> lr).  optax-compatible.
    lr_schedule:    Optional[Callable[[int], float]] = field(default=None, repr=False)


_SOLVERS = {
    'tsit5':    diffrax.Tsit5,
    'dopri5':   diffrax.Dopri5,
    'kvaerno5': diffrax.Kvaerno5,
}
_ADJOINTS = {
    'recursive_checkpoint': diffrax.RecursiveCheckpointAdjoint,
    'backsolve':            diffrax.BacksolveAdjoint,
    'direct':               diffrax.DirectAdjoint,
}


# ---------------------------------------------------------------------------
# Core integration helper
# ---------------------------------------------------------------------------

def _make_integrator(problem, cfg, in_mean, in_std, out_mean, out_std):
    """Return a function integrate(mlp, x0, t_save, consts) -> (ys, ok)."""
    t0 = float(problem.t_span[0])
    t1 = float(problem.t_span[1])
    solver  = _SOLVERS[cfg.solver]()
    adjoint = _ADJOINTS[cfg.adjoint]()
    controller = diffrax.PIDController(rtol=cfg.rtol, atol=cfg.atol)

    def integrate(mlp, x0, t_save, consts):
        def field(t, x, args):
            norm_x = (x - in_mean) / in_std
            z = out_mean + out_std * mlp(norm_x)
            return problem.rhs_jax(t, x, z, args)

        term = diffrax.ODETerm(field)
        sol = diffrax.diffeqsolve(
            term, solver, t0=t0, t1=t1, dt0=None, y0=x0, args=consts,
            saveat=diffrax.SaveAt(ts=t_save),
            stepsize_controller=controller,
            adjoint=adjoint,
            max_steps=cfg.max_steps,
            throw=False,
        )
        ok = (sol.result == diffrax.RESULTS.successful)
        return sol.ys, ok

    return integrate


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_sequential(
    problem: ProblemDefinition,
    mlp: SimpleMLP,
    cfg: SequentialConfig,
    data: InstanceData,
    key: jax.Array = jax.random.PRNGKey(0),
) -> Tuple[SimpleMLP, dict]:
    """
    Train a neural network via the standard sequential approach (forward
    integration + Adam).

    Pretraining / smoothing are handled externally (same pipeline as the other
    methods).  ``data`` is used only for normalization statistics so the result
    is comparable to the simultaneous / decomposition methods.

    Returns
    -------
    mlp     : SimpleMLP  (best data-fit weights)
    history : dict with keys obj_history, data_fit_history, penalty_history,
              grad_norm_history, min_constraint_history, max_violation_history,
              wall_step_history, eval_steps, integration_failures
    """
    jax.config.update("jax_enable_x64", True)

    if problem.obs_times is None or problem.obs_values is None:
        raise RuntimeError("problem.obs_times / obs_values not set — call generate_data first.")
    if problem.obs_dim != problem.input_dim:
        raise NotImplementedError(
            "train_sequential assumes observed variables are the (full) state "
            f"(obs_dim == input_dim); got obs_dim={problem.obs_dim}, "
            f"input_dim={problem.input_dim}."
        )

    # Normalization statistics (shared with all other methods).
    in_mean  = jnp.array(data.input_mean)
    in_std   = jnp.array(data.input_std)
    out_mean = jnp.array(data.output_mean)
    out_std  = jnp.array(data.output_std)

    n_traj = problem.num_trajectories
    t0, t1 = float(problem.t_span[0]), float(problem.t_span[1])

    # Per-trajectory static data.
    x0_list     = [jnp.array(problem.ics[i], dtype=float)        for i in range(n_traj)]
    consts_list = [problem.rhs_traj_consts(i)                    for i in range(n_traj)]
    tobs_list   = [jnp.array(problem.obs_times[i],  dtype=float) for i in range(n_traj)]
    yobs_list   = [jnp.array(problem.obs_values[i], dtype=float) for i in range(n_traj)]

    # Warmup (curriculum) obs subsets.
    t_warm_end = t0 + cfg.warmup_frac * (t1 - t0)
    tobs_warm, yobs_warm = [], []
    for i in range(n_traj):
        mask = np.asarray(problem.obs_times[i]) <= t_warm_end
        if mask.sum() < 2:                      # keep at least 2 points
            mask[:2] = True
        tobs_warm.append(jnp.array(np.asarray(problem.obs_times[i])[mask], dtype=float))
        yobs_warm.append(jnp.array(np.asarray(problem.obs_values[i])[mask], dtype=float))

    # Dense grid (per trajectory: same uniform grid) for constraint diagnostics.
    t_dense = jnp.linspace(t0, t1, cfg.n_dense)

    integrate = _make_integrator(problem, cfg, in_mean, in_std, out_mean, out_std)

    # --- loss factory (one closure per obs grid: full and warmup) ----------
    def make_loss(tobs, yobs):
        @eqx.filter_jit
        def loss_fn(m):
            mse_total = 0.0
            pen_total = 0.0
            for i in range(n_traj):
                ys, _ = integrate(m, x0_list[i], tobs[i], consts_list[i])
                mse_total += jnp.mean((ys - yobs[i]) ** 2)
                if cfg.penalty_coef > 0.0:
                    # Path constraints are analytic prior knowledge evaluable
                    # anywhere — penalise them on a DENSE, full-horizon grid
                    # (not just at observation times), giving the penalty-based
                    # baseline its best shot at satisfying the constraint over
                    # the whole trajectory, including any extrapolation region.
                    ysd, _ = integrate(m, x0_list[i], t_dense, consts_list[i])
                    g = jax.vmap(
                        lambda tt, xx: problem.path_constraint_residuals_jax(
                            tt, xx, out_mean + out_std * m((xx - in_mean) / in_std),
                            consts_list[i],
                        )
                    )(t_dense, ysd)
                    pen_total += jnp.mean(jax.nn.relu(-g) ** 2)
            mse = mse_total / n_traj
            pen = pen_total / n_traj
            reg = _l2_params(m)
            obj = mse + cfg.penalty_coef * pen + cfg.param_reg_coef * reg
            return obj, (mse, pen)
        return loss_fn

    loss_full = make_loss(tobs_list, yobs_list)
    loss_warm = make_loss(tobs_warm, yobs_warm) if cfg.warmup_steps > 0 else loss_full

    # --- dense constraint diagnostics (not differentiated) -----------------
    @eqx.filter_jit
    def dense_constraint_stats(m):
        min_g = jnp.inf
        max_v = 0.0
        for i in range(n_traj):
            ys, _ = integrate(m, x0_list[i], t_dense, consts_list[i])
            g = jax.vmap(
                lambda tt, xx: problem.path_constraint_residuals_jax(
                    tt, xx, out_mean + out_std * m((xx - in_mean) / in_std),
                    consts_list[i],
                )
            )(t_dense, ys)
            if g.shape[-1] > 0:
                min_g = jnp.minimum(min_g, jnp.min(g))
                max_v = jnp.maximum(max_v, jnp.max(jax.nn.relu(-g)))
        return min_g, max_v

    has_constraints = problem.path_constraint_residuals_jax(
        t0, x0_list[0],
        out_mean + out_std * mlp((x0_list[0] - in_mean) / in_std),
        consts_list[0],
    ).shape[-1] > 0

    # --- optimizer ---------------------------------------------------------
    sched = cfg.lr_schedule if cfg.lr_schedule is not None else optax.constant_schedule(cfg.lr)
    if np.isfinite(cfg.grad_clip_norm):
        optimizer = optax.chain(optax.clip_by_global_norm(cfg.grad_clip_norm), optax.adam(sched))
    else:
        optimizer = optax.adam(sched)
    opt_state = optimizer.init(eqx.filter(mlp, eqx.is_array))

    @eqx.filter_jit
    def make_step(m, state, loss_fn):
        (obj, (mse, pen)), grad = eqx.filter_value_and_grad(loss_fn, has_aux=True)(m)
        gnorm = optax.global_norm(eqx.filter(grad, eqx.is_array))
        updates, state = optimizer.update(grad, state, m)
        m = eqx.apply_updates(m, updates)
        return m, state, obj, mse, pen, gnorm

    # --- training loop -----------------------------------------------------
    logger.info(f"=== Sequential training ({cfg.n_steps} steps, penalty_coef={cfg.penalty_coef:.1e}) ===")
    obj_h, mse_h, pen_h, gnorm_h, wall_h = [], [], [], [], []
    min_c_h, max_v_h, eval_steps = [], [], []
    n_fail = 0

    best_mlp, best_mse = mlp, np.inf
    stagnation = 0

    for step in range(1, cfg.n_steps + 1):
        t_start = time.perf_counter()
        loss_fn = loss_warm if (cfg.warmup_steps > 0 and step <= cfg.warmup_steps) else loss_full
        mlp, opt_state, obj, mse, pen, gnorm = make_step(mlp, opt_state, loss_fn)
        obj = float(obj); mse = float(mse); pen = float(pen); gnorm = float(gnorm)
        wall_h.append(time.perf_counter() - t_start)

        if not np.isfinite(obj):
            n_fail += 1

        obj_h.append(obj); mse_h.append(mse); pen_h.append(pen); gnorm_h.append(gnorm)

        if np.isfinite(mse) and mse < best_mse:
            best_mse, best_mlp = mse, mlp
            stagnation = 0
        else:
            stagnation += 1

        if has_constraints and (step % cfg.eval_every == 0 or step == 1):
            min_g, max_v = dense_constraint_stats(mlp)
            min_c_h.append(float(min_g)); max_v_h.append(float(max_v)); eval_steps.append(step)

        if step % cfg.log_every == 0 or step == 1:
            extra = ""
            if has_constraints and min_c_h:
                extra = f"  min_g={min_c_h[-1]:.3e}"
            logger.info(
                f"  step {step:5d}  obj={obj:.4e}  mse={mse:.4e}  "
                f"pen={pen:.3e}  |grad|={gnorm:.2e}{extra}"
            )

        if cfg.patience > 0 and stagnation >= cfg.patience:
            logger.info(f"  Early stopping at step {step}: data-fit stagnated {cfg.patience} steps.")
            break

    logger.info(f"=== Sequential training complete (best mse={best_mse:.4e}, integ. failures={n_fail}) ===")

    history = {
        'obj_history':            obj_h,
        'data_fit_history':       mse_h,
        'penalty_history':        pen_h,
        'grad_norm_history':      gnorm_h,
        'wall_step_history':      wall_h,
        'min_constraint_history': min_c_h,
        'max_violation_history':  max_v_h,
        'eval_steps':             eval_steps,
        'integration_failures':   n_fail,
    }
    return best_mlp, history


# ---------------------------------------------------------------------------
# Post-training: forward-integrate to an InstanceData (for plotting / test MSE)
# ---------------------------------------------------------------------------

def integrate_to_instance_data(
    problem: ProblemDefinition,
    mlp: SimpleMLP,
    data: InstanceData,
    cfg: Optional[SequentialConfig] = None,
    t_grid: Optional[np.ndarray] = None,
    ics: Optional[np.ndarray] = None,
) -> InstanceData:
    """
    Forward-integrate the trained NN ODE on each trajectory and return an
    InstanceData (drop-in for ``extract_instance_data`` so the existing plotting
    pipeline works).

    Parameters
    ----------
    ics : optional (n, state_dim) array of initial conditions to integrate from.
          Defaults to ``problem.ics`` (training ICs).  Pass held-out ICs to
          evaluate generalization / test MSE.
    """
    jax.config.update("jax_enable_x64", True)
    cfg = cfg or SequentialConfig()
    in_mean  = jnp.array(data.input_mean);  in_std  = jnp.array(data.input_std)
    out_mean = jnp.array(data.output_mean); out_std = jnp.array(data.output_std)

    ics = problem.ics if ics is None else np.asarray(ics)
    t0, t1 = float(problem.t_span[0]), float(problem.t_span[1])
    if t_grid is None:
        t_grid = np.linspace(t0, t1, 200)
    t_save = jnp.array(t_grid, dtype=float)

    integrate = _make_integrator(problem, cfg, in_mean, in_std, out_mean, out_std)

    def z_of_x(x):
        return out_mean + out_std * mlp((x - in_mean) / in_std)

    trajectories: List[TrajectoryData] = []
    for i in range(len(ics)):
        consts = problem.rhs_traj_consts(i if i < problem.num_trajectories else 0)
        ys, ok = integrate(mlp, jnp.array(ics[i], dtype=float), t_save, consts)
        if not ok:
            logger.warning(f"integrate_to_instance_data: trajectory {i} integration failed.")
        x_pred = np.asarray(ys)
        z_pred = np.asarray(jax.vmap(z_of_x)(ys))
        trajectories.append(TrajectoryData(
            sampling_times=np.asarray(t_grid),
            nn_input=x_pred,
            nn_output=z_pred,
            obs=x_pred,
            aux_vars=None,
        ))
    return InstanceData(trajectories)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _l2_params(mlp) -> jax.Array:
    """Mean squared NN parameter value (L2 regularisation term)."""
    leaves = [w for w in jax.tree_util.tree_leaves(eqx.filter(mlp, eqx.is_array))]
    if not leaves:
        return jnp.array(0.0)
    sq = sum(jnp.sum(w ** 2) for w in leaves)
    n  = sum(w.size for w in leaves)
    return sq / n
