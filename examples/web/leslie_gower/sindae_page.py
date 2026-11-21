"""The SiNDAE Leslie-Gower page's three steps: generate the data, train the
network, and infer with it. Each step takes its options as JSON text and
gives back what the page draws as JSON text. The steps follow SiNDAE's
examples/leslie_gower.py with METHOD = 'simul' and USE_GBM = False, through
SiNDAE's own functions: generate_data, solve_smoother,
extract_instance_data, build_simultaneous_model and the network of
nn_block.py written as Pyomo algebra. The inference is SiNDAE's
make_inference_model with the network written the same way in place of its
grey box.

Pyodide has no build of JAX, so this file builds the network and pretrains
it in NumPy, and gives SiNDAE's functions PounceInProcess as their NLP
backend, pounce's Python package solving each model in this process.

The standard errors come from pounce's sensitivity of the solved NLPs. In
training, the measurements enter the objective as mutable Params, so dθ/dy,
the derivative of the trained weights with respect to the measurements, gives
Cov(θ) = (dθ/dy) Σ_y (dθ/dy)ᵀ, with Σ_y the variances of the noise step 1
adds. The same derivatives of the trained states give their variances. In the
inference, the weights are declared as parameters, and dx/dθ gives
Var(x) = (dx/dθ) Cov(θ) (dx/dθ)ᵀ. Each is first order in the noise, and holds
fixed the standardization constants of the data fit and of the network, which
SiNDAE computes from the measurements and from the smoother's solution.
"""
import json
import math
import time

import numpy as np
import pyomo.environ as pyo
from pyomo.util.calc_var_value import calculate_variable_from_constraint
from pyomo_pounce import declare_sens_param, sens_jacobian

from sindae.algorithms.model_builder_utils import NORM_INPUT_NAME, NORM_OBS_NAME, NORM_OUTPUT_NAME, add_norm_vars_to_block, add_normalization_to_block
from sindae.algorithms.simultaneous.model_builder import build_simultaneous_model
from sindae.algorithms.simultaneous.nn_block import NNBlock, make_nn_output_expression
from sindae.algorithms.smoother import solve_smoother
from sindae.data_utils import extract_instance_data, generate_data
from sindae.example_problems import LeslieGowerProblem
from sindae.solvers import NLPResult, NLPSolver

# The worker sets PROGRESS to a function that sends one row of the
# training solve, as JSON text, to the page while pounce solves.
PROGRESS = None


def _sigmoid(x):
    # 1/(1 + exp(-x)) written with tanh, which does not overflow for large -x.
    return 0.5 * (1 + np.tanh(0.5 * x))


def activate(name, x):
    """The activation `name` of nn_block.py's _act_str2pyo and its derivative, at x."""
    if name == "softplus":
        return np.logaddexp(0, x), _sigmoid(x)
    if name == "tanh":
        y = np.tanh(x)
        return y, 1 - y * y
    s = _sigmoid(x)
    return x * s, s + x * s * (1 - s)


class Layer:
    def __init__(self, n_in, n_out, rng):
        lim = 1 / np.sqrt(n_in)
        self.weight = rng.uniform(-lim, lim, size=(n_out, n_in))
        self.bias = rng.uniform(-lim, lim, size=n_out)


class NumpyMLP:
    """SiNDAE's SimpleMLP in NumPy, with the attributes eqx_mlp_to_nn_block
    reads and the activations by name. The weights are drawn as Equinox's
    Linear draws them, uniform on ±1/sqrt(inputs)."""

    def __init__(self, in_size, out_size, widths, activations, seed=0):
        rng = np.random.default_rng(seed)
        self.in_size, self.out_size = in_size, out_size
        self.widths, self.activations = list(widths), list(activations)
        sizes = [in_size, *widths, out_size]
        self.layers = [Layer(sizes[k], sizes[k + 1], rng) for k in range(len(sizes) - 1)]


def pretrain(mlp, data, epochs=200, batch_size=32, reg_coef=1e-3, lr=1e-3, seed=0):
    """SiNDAE's pretrain_mlp in NumPy: fits mlp to data's normalized (input,
    output) pairs as train_eqx_mlp does, by Adam on the mean squared error
    plus reg_coef times the mean square of each array of weights and biases."""
    x = (np.vstack(data.nn_input) - data.input_mean) / data.input_std
    y = (np.vstack(data.nn_output) - data.output_mean) / data.output_std
    params = [p for layer in mlp.layers for p in (layer.weight, layer.bias)]
    m1 = [np.zeros_like(p) for p in params]
    m2 = [np.zeros_like(p) for p in params]
    rng = np.random.default_rng(seed)
    last = len(mlp.layers) - 1
    step = 0
    for _ in range(epochs):
        order = rng.permutation(len(x))
        for start in range(0, len(x), batch_size):
            idx = order[start : start + batch_size]
            acts, slopes = [x[idx]], []
            for k, layer in enumerate(mlp.layers):
                pre = acts[-1] @ layer.weight.T + layer.bias
                if k < last:
                    out, slope = activate(mlp.activations[k], pre)
                else:
                    out, slope = pre, None
                acts.append(out)
                slopes.append(slope)
            grad = 2 * (acts[-1] - y[idx]) / y[idx].size
            grads = []
            for k in reversed(range(len(mlp.layers))):
                layer = mlp.layers[k]
                if k < last:
                    grad = grad * slopes[k]
                gw = grad.T @ acts[k] + reg_coef * 2 * layer.weight / layer.weight.size
                gb = grad.sum(0) + reg_coef * 2 * layer.bias / layer.bias.size
                grads[:0] = [gw, gb]
                grad = grad @ layer.weight
            step += 1
            for p, g, a, b in zip(params, grads, m1, m2):
                a *= 0.9
                a += 0.1 * g
                b *= 0.999
                b += 0.001 * g * g
                p -= lr * (a / (1 - 0.9**step)) / (np.sqrt(b / (1 - 0.999**step)) + 1e-8)
    return mlp


class PounceInProcess(NLPSolver):
    """pounce's Python package as SiNDAE's NLP backend. pyomo-pounce solves the
    model in this process, and keeps the solve's KKT factorization for
    sens_jacobian where the model declares parameters. `watch`, where given,
    is called at each of pounce's iterations with pyomo-pounce's Iteration."""

    name = "pounce"

    def __init__(self, options=None, watch=None):
        self._options = dict(options or {})
        self.watch = watch

    def solve(self, model, *, tee=False, extra_options=None, return_nlp=False):
        if return_nlp:
            raise ValueError("PounceInProcess gives back no NLP")
        options = {**self._options, **(extra_options or {})}
        started = time.perf_counter()
        results = pyo.SolverFactory("pounce").solve(model, options=options, intermediate=self.watch)
        message = str(results.solver.message or results.solver.termination_condition)
        timing = {
            "solve": time.perf_counter() - started,
            "iterations": results.solver.statistics.black_box.number_of_iterations,
            "status": message.split(": ", 1)[-1],
        }
        return NLPResult(result=results, timing=timing)


def _derivatives(targets, params):
    """d(target)/d(param) at the solution, as an array with a row for each
    target and a column for each param: one backsolve with the solve's KKT
    factorization for each param. A target fixed in the solve has a row of
    zeros."""
    from pyomo_pounce.sens import Jacobian, _param_pin, _session_for

    session = _session_for(params[0])
    g = Jacobian(session, targets, params)
    rows = [None if td.fixed else g._entry(td) for td in targets]
    out = np.zeros((len(targets), len(params)))
    for c, pd in enumerate(params):
        column = session.column(_param_pin(session, pd))
        for r, row in enumerate(rows):
            if row is not None:
                out[r, c] = column[row]
    return out


def _weights(nn):
    """The network's weights and biases, layer by layer, each layer's weights
    row by row and then its biases, the order of the page's plots."""
    out = []
    for layer in nn.layers:
        out += [layer.weight[i, j] for i in range(layer.out_features) for j in range(layer.in_features)]
        out += [layer.bias[i] for i in range(layer.out_features)]
    return out


def _by_layer(nn, flat):
    """A list in _weights' order as [[weights rows], [biases]] for each layer."""
    out, at = [], 0
    for layer in nn.layers:
        w = [list(flat[at + i * layer.in_features : at + (i + 1) * layer.in_features]) for i in range(layer.out_features)]
        at += layer.out_features * layer.in_features
        out.append([w, list(flat[at : at + layer.out_features])])
        at += layer.out_features
    return out


def _add_layers(m, nn):
    """Adds each layer's Block of SiNDAE's NNBlock to the model m. NNBlock
    makes each layer's Vars on a Block of its own that no model holds, so
    every layer's Vars share the names weight[i,j] and bias[i]. pyomo-pounce
    finds each Var of the solve by its name, and on the model they are
    nn_layer_0.weight[i,j] and so on."""
    for li, layer in enumerate(nn.layers):
        m.add_component(f"nn_layer_{li}", layer.block)


def _measured_objective(m, problem, reg_coef):
    """build_simultaneous_model's objective written again, with each target an
    expression of the measurements, an indexed mutable Param, in place of a
    number. Each measurement enters through one equality with a Var. At the
    measured values the objective equals SiNDAE's."""
    obs_dim = len(problem.obs_mean)
    mean, std = np.asarray(problem.obs_mean), np.asarray(problem.obs_std)
    keys = [(k, i, j) for k in range(problem.num_trajectories) for i in range(len(problem.obs_times[k])) for j in range(obs_dim)]
    measured = {key: float(np.asarray(problem.obs_values[key[0]])[key[1], key[2]]) for key in keys}
    m.measured = pyo.Param(keys, mutable=True, initialize=measured)
    m.measured_value = pyo.Var(keys, initialize=measured)
    m.measured_pin = pyo.Constraint(keys, rule=lambda mo, k, i, j: mo.measured_value[k, i, j] == mo.measured[k, i, j])
    fit = 0
    for k in range(problem.num_trajectories):
        block = m.trajectories[k]
        times = sorted(block.t)
        t_obs = np.asarray(problem.obs_times[k], dtype=float)
        # Linear interpolation, as _compute_norm_targets does with np.interp: column i is the weight of measurement i at each time.
        weight = np.column_stack([np.interp(times, t_obs, np.eye(len(t_obs))[i]) for i in range(len(t_obs))])
        norm_obs = getattr(block, NORM_OBS_NAME)
        terms = []
        for ti, t in enumerate(times):
            near = np.nonzero(weight[ti])[0]
            for j in range(obs_dim):
                target = (sum(float(weight[ti, i]) * m.measured_value[k, int(i), j] for i in near) - float(mean[j])) / float(std[j])
                terms.append((norm_obs[t, j] - target) ** 2)
        fit += pyo.quicksum(terms) / (len(times) * obs_dim)
    reg = pyo.quicksum(v**2 for v in _weights(m._nn_block))
    m.obj.deactivate()
    m.measured_obj = pyo.Objective(expr=fit + reg_coef * reg)
    theirs, ours = pyo.value(m.obj), pyo.value(m.measured_obj)
    assert abs(theirs - ours) <= 1e-9 * (1 + abs(theirs)), (theirs, ours)
    return keys


NCP = 3
STATE = {}


def _trajectories(data, problem=None):
    """Each trajectory's times, prey, predator and z, and its measurements where `problem` has them."""
    out = []
    for k in range(data.num_trajectories):
        traj = data[k]
        row = {
            "t": traj.sampling_times.tolist(),
            "prey": traj.nn_input[:, 0].tolist(),
            "predator": traj.nn_input[:, 1].tolist(),
            "z": traj.nn_output[:, 0].tolist(),
        }
        if problem is not None and problem.obs_times is not None:
            row["obs_t"] = np.asarray(problem.obs_times[k]).tolist()
            row["obs_prey"] = np.asarray(problem.obs_values[k])[:, 0].tolist()
            row["obs_predator"] = np.asarray(problem.obs_values[k])[:, 1].tolist()
        out.append(row)
    return out


def _true_z(params, prey, predator):
    """The true term of LeslieGowerProblem, z0 = r2 - a2 x1/x0."""
    return params["r2"] - params["a2"] * np.asarray(predator) / np.asarray(prey)


def generate(options):
    """Step 1: SiNDAE's generate_data on the true model from each initial condition, with noise added to every Nth point."""
    o = json.loads(options)
    started = time.perf_counter()
    problem = LeslieGowerProblem(ics=np.array(o["ics"], dtype=float), params=dict(o["params"]), t_span=(0.0, o["t_end"]), nfe=o["nfe"], ncp=NCP)
    true = generate_data(problem=problem, noise_std=np.array(o["noise"], dtype=float), obs_every=o["obs_every"], seed=o["seed"], nlp_solver=PounceInProcess())
    if true is None:
        STATE.clear()
        return json.dumps({"ok": False, "message": "pounce found no solution of the true model."})
    STATE.clear()
    STATE.update(problem=problem, params=dict(o["params"]), true=true, noise=list(o["noise"]))
    return json.dumps({"ok": True, "ms": 1000 * (time.perf_counter() - started), "trajectories": _trajectories(true, problem)})


def train(options):
    """Step 2: SiNDAE's smoother, the pretraining, and the simultaneous NLP with the network as Pyomo algebra."""
    o = json.loads(options)
    problem = STATE["problem"]
    times = {}
    clock = time.perf_counter()
    problem.nfe, problem.ncp = o["nfe"], NCP
    # SiNDAE's Lyapunov condition, dV/dt <= 0 at every collocation time after the first, in the smoother and so in the training NLP built on it.
    problem.lyap_descent = bool(o.get("lyapunov", False))
    mlp = NumpyMLP(problem.input_dim, problem.z_dim, o["widths"], [o["activation"]] * len(o["widths"]), seed=o["seed"])
    smoother_m = solve_smoother(problem, mlp, smooth_coef=o["smooth_coef"], nlp_solver=PounceInProcess())
    times["smoother"] = smoother_m._pounce_timing
    if smoother_m._solver_result.solver.termination_condition != pyo.TerminationCondition.optimal:
        return json.dumps({"ok": False, "message": f"pounce ended the smoother {times['smoother']['status']}."})
    smoother_data = extract_instance_data(problem, smoother_m)
    now = time.perf_counter()
    times["smoother_ms"] = 1000 * (now - clock)
    clock = now
    pretrain(mlp, smoother_data, epochs=o["epochs"], batch_size=32, reg_coef=o["reg_coef"], seed=o["seed"])
    pretrained = [[layer.weight.tolist(), layer.bias.tolist()] for layer in mlp.layers]
    now = time.perf_counter()
    times["pretrain_ms"] = 1000 * (now - clock)
    clock = now
    m = build_simultaneous_model(problem, mlp, list(range(problem.num_trajectories)), smoother_data, smoother_model=smoother_m, reg_coef=o["reg_coef"])
    now = time.perf_counter()
    times["build_ms"] = 1000 * (now - clock)
    clock = now
    _add_layers(m, m._nn_block)
    keys = _measured_objective(m, problem, o["reg_coef"])
    declare_sens_param(m.measured)
    weights = _weights(m._nn_block)
    # Each iteration's row goes to the page as pounce solves, with the weights where pounce is not in restoration.
    last = [-1]

    def watch(it):
        if PROGRESS is None:
            return
        if it.iteration < last[0]:
            PROGRESS(json.dumps({"resolve": True}))
        last[0] = it.iteration
        row = {"iter": it.iteration, "restoration": bool(it.restoration), "objective": it.objective, "inf_pr": it.inf_pr, "inf_du": it.inf_du, "lg_mu": math.log10(it.mu) if it.mu > 0 else None}
        if it.values is not None:
            row["weights"] = _by_layer(m._nn_block, [float(it.values[v]) for v in weights])
        if all(math.isfinite(row[name]) for name in ("objective", "inf_pr", "inf_du")):
            PROGRESS(json.dumps(row))

    solved = PounceInProcess({"tol": 1e-6, "max_iter": 1000, "hessian_approximation": o["hessian"]}, watch=watch).solve(m)
    times["training"] = solved.timing
    times["training_ms"] = 1000 * (time.perf_counter() - clock)
    if solved.result.solver.termination_condition not in (pyo.TerminationCondition.optimal, pyo.TerminationCondition.locallyOptimal):
        return json.dumps({"ok": False, "message": f"pounce ended training {solved.timing['status']}.", "times": times})
    trained = extract_instance_data(problem, m)
    nn = m._nn_block
    # The derivatives come from the KKT factorization of the training solve,
    # which after an L-BFGS solve holds L-BFGS's Hessian. The page notes that
    # the standard errors may then be inaccurate.
    clock = time.perf_counter()
    uq = _training_uncertainty(m, problem, keys, weights)
    times["uncertainty_ms"] = 1000 * (time.perf_counter() - clock)
    STATE["network"] = {
        "widths": list(o["widths"]),
        "activations": [o["activation"]] * len(o["widths"]),
        "weights": [(np.array(layer._w), np.array(layer._b)) for layer in nn.layers],
        "stats": (smoother_data.input_mean, smoother_data.input_std, smoother_data.output_mean, smoother_data.output_std),
        "cov": uq["cov"],
        # The least and greatest prey and predator of the training data's
        # initial conditions and the trained states, the states the network
        # was fitted at. The fit leaves each trajectory's first point free,
        # so the initial conditions can lie a little past the trained states.
        "range": np.array([
            np.min(np.vstack([*trained.nn_input, problem.ics]), axis=0),
            np.max(np.vstack([*trained.nn_input, problem.ics]), axis=0),
        ]),
    }
    STATE.pop("simulated", None)
    rows = _trajectories(trained)
    for k, row in enumerate(rows):
        row["true_z"] = _true_z(STATE["params"], row["prey"], row["predator"]).tolist()
        row["sd"] = uq["sd"][k]
    weights = {
        "pretrained": pretrained,
        "trained": [[w.tolist(), b.tolist()] for w, b in STATE["network"]["weights"]],
        "se": _by_layer(nn, np.sqrt(np.maximum(np.diag(uq["cov"]), 0.0)).tolist()),
    }
    # The fit to the measurements of every trajectory together, at their own
    # times, each state's trained value read from its collocation
    # polynomial, and the learned z0 against the true one at the trained
    # states.
    fit = {}
    for j, name in enumerate(("prey", "predator")):
        measured = np.concatenate([np.asarray(problem.obs_values[k])[:, j] for k in range(problem.num_trajectories)])
        fitted = np.concatenate([[_collocation_value(trained[k].sampling_times, trained[k].nn_input[:, j], t) for t in problem.obs_times[k]] for k in range(problem.num_trajectories)])
        fit[name] = _fit(measured, fitted)
    fit["z"] = _fit(np.concatenate([row["true_z"] for row in rows]), np.concatenate([row["z"] for row in rows]))
    penalty = sum(float(np.sum(w**2) + np.sum(b**2)) for w, b in STATE["network"]["weights"])
    objective = pyo.value(m.obj)
    metrics = {"fit": fit, "objective": objective, "penalty": o["reg_coef"] * penalty, "data_fit": objective - o["reg_coef"] * penalty}
    return json.dumps({"ok": True, "times": times, "trajectories": rows, "smoother": _trajectories(smoother_data), "weights": weights, "metrics": metrics})


def _training_uncertainty(m, problem, keys, weights):
    """Cov(θ) = G Σ_y Gᵀ with G = dθ/dy, and the standard deviation of each
    trained state and of z0 at every collocation time, from the training
    solve's derivatives with respect to the measurements y."""
    params = [m.measured[key] for key in keys]
    noise = np.asarray(STATE["noise"], dtype=float)
    variance = np.array([noise[j] ** 2 for (_k, _i, j) in keys])
    g_theta = _derivatives(weights, params)
    cov = (g_theta * variance) @ g_theta.T
    sd = []
    for k in range(problem.num_trajectories):
        block = m.trajectories[k]
        times = sorted(block.t)
        out = {}
        for name, targets in (
            ("prey", [block.x[t, 0] for t in times]),
            ("predator", [block.x[t, 1] for t in times]),
            ("z", [block.z[t, 0] for t in times]),
        ):
            g = _derivatives(targets, params)
            out[name] = np.sqrt((g * g) @ variance).tolist()
        sd.append(out)
    return {"cov": cov, "sd": sd}


def _fit(measured, fitted):
    """The RMSE of `fitted` against `measured`, and R² = 1 - SS_res/SS_tot about the mean of `measured`."""
    measured, fitted = np.asarray(measured, dtype=float), np.asarray(fitted, dtype=float)
    residual = float(np.sum((measured - fitted) ** 2))
    spread = float(np.sum((measured - measured.mean()) ** 2))
    return {"rmse": float(np.sqrt(residual / len(measured))), "r2": 1 - residual / spread if spread > 0 else None}


def _inside(truths, values, sds):
    """The share of the ground truth's values within ±2 standard deviations of the trained model's."""
    truths, values, sds = (np.asarray(a, dtype=float) for a in (truths, values, sds))
    return float(np.mean(np.abs(truths - values) <= 2 * sds))


def _collocation_value(times, values, t):
    """The collocation polynomial's value at `t`: the Lagrange polynomial through the start and the NCP Radau points of the element t lies in."""
    times, values = np.asarray(times, dtype=float), np.asarray(values, dtype=float)
    elements = (len(times) - 1) // NCP
    e = min(max(int(np.searchsorted(times[::NCP], t, side="right")) - 1, 0), elements - 1)
    nodes, ys = times[NCP * e : NCP * e + NCP + 1], values[NCP * e : NCP * e + NCP + 1]
    total = 0.0
    for i, (ti, yi) in enumerate(zip(nodes, ys)):
        basis = 1.0
        for k, tk in enumerate(nodes):
            if k != i:
                basis *= (t - tk) / (ti - tk)
        total += yi * basis
    return total


def _network_block(network):
    """The trained network as SiNDAE's NNBlock with its weights fixed."""
    w0 = network["weights"]
    block = NNBlock(input_dim=w0[0][0].shape[1], output_dim=w0[-1][0].shape[0], widths=network["widths"], activations=network["activations"])
    for layer, (w, b) in zip(block.layers, w0):
        layer.weight = w
        layer.bias = b
    block.fix()
    return block


def _network_z(network, x):
    """The trained network's z0 at the states x, in the units of z0."""
    in_mean, in_std, out_mean, out_std = network["stats"]
    h = (np.asarray(x, dtype=float) - in_mean) / in_std
    last = len(network["weights"]) - 1
    for k, (w, b) in enumerate(network["weights"]):
        h = w @ h + b
        if k < last:
            h = activate(network["activations"][k], h)[0]
    return float(h[0] * out_std[0] + out_mean[0])


def _start_from_integration(block, ic, params, network, substeps=20):
    """Starts pounce from the hybrid model integrated by the Runge-Kutta
    method of order 4 through the collocation times. Gives back the time a
    state passes 1e3, or None where none does."""
    p = params

    def rate(x):
        z = _network_z(network, x)
        return np.array([x[0] * (p["r1"] - p["a1"] * x[1] - p["b1"] * x[0]), x[1] * z])

    times = sorted(block.t)
    x = np.array(ic, dtype=float)
    in_mean, in_std, out_mean, out_std = network["stats"]
    leaves = None
    for n, t in enumerate(times):
        if n > 0 and leaves is None:
            h = (t - times[n - 1]) / substeps
            for k in range(substeps):
                k1 = rate(x)
                k2 = rate(x + h / 2 * k1)
                k3 = rate(x + h / 2 * k2)
                k4 = rate(x + h * k3)
                step = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
                # Where a state passes 1e3 or is no longer finite, the states stay at their last values to the end.
                if not (np.all(np.isfinite(step)) and np.all(np.abs(step) < 1e3)):
                    leaves = times[n - 1] + (k + 1) * h
                    break
                x = np.maximum(step, 1e-8)
        z = _network_z(network, x)
        dx = rate(x)
        for j in range(2):
            if not block.x[t, j].fixed:
                block.x[t, j].set_value(x[j])
            block.dxdt[t, j].set_value(dx[j])
            getattr(block, NORM_INPUT_NAME)[t, j].set_value((x[j] - in_mean[j]) / in_std[j])
        block.z[t, 0].set_value(z)
        getattr(block, NORM_OUTPUT_NAME)[t, 0].set_value((z - out_mean[0]) / out_std[0])
        calculate_variable_from_constraint(block.lyap_var[t], block.lyap_constr[t])
    return leaves


def _leaves(ic, params, network, t_end, steps=4000):
    """The hybrid model integrated from `ic`: the time a state passes 1e3,
    and the time a state first leaves the range of the trained states,
    each None where it does not happen before t_end."""
    p = params
    low, high = network["range"]
    x = np.array(ic, dtype=float)
    h = t_end / steps
    outside = 0.0 if np.any(x < low) or np.any(x > high) else None
    for k in range(steps):
        rate = lambda y: np.array([y[0] * (p["r1"] - p["a1"] * y[1] - p["b1"] * y[0]), y[1] * _network_z(network, y)])
        k1 = rate(x)
        k2 = rate(x + h / 2 * k1)
        k3 = rate(x + h / 2 * k2)
        x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + rate(x + h * k3))
        if not (np.all(np.isfinite(x)) and np.all(np.abs(x) < 1e3)):
            return (k + 1) * h, outside
        if outside is None and (np.any(x < low) or np.any(x > high)):
            outside = (k + 1) * h
    return None, outside


def _hybrid(ic, params, network, t_end, nfe):
    """SiNDAE's make_inference_model for one initial condition, with the
    trained network written as Pyomo algebra in place of its grey box,
    solved by pounce. Where the integrated model's states pass 1e3, the
    model ends at the last element's end before that time."""
    leaves, outside = _leaves(ic, params, network, t_end)
    found = {"leaves": leaves, "outside": outside, "t_end": t_end}
    if leaves is not None:
        element = t_end / nfe
        nfe = int(leaves // element)
        found["t_end"] = nfe * element
        if nfe < 1:
            return None, {"status": "not solved", "iterations": 0, "solve": 0.0, "write": 0.0, "nl_bytes": 0}, found
        t_end = nfe * element
    problem = LeslieGowerProblem(ics=np.array([ic], dtype=float), params=dict(params), t_span=(0.0, t_end), nfe=nfe, ncp=NCP)
    in_mean, in_std, out_mean, out_std = network["stats"]
    m = pyo.ConcreteModel()
    m.traj_set = pyo.RangeSet(0, 0)
    m.trajectories = pyo.Block(m.traj_set)
    block = m.trajectories[0]
    problem.build_trajectory(block, 0)
    add_norm_vars_to_block(block, block.t, problem.input_dim, problem.z_dim)
    add_normalization_to_block(block, problem.get_input_vars, problem.get_output_vars, in_mean, in_std, out_mean, out_std, block.t)
    problem.discretize(m)
    # The weights are fixed Vars on the layers' own Blocks, as build_simultaneous_model leaves them, so the NL writer writes them as numbers.
    nn = _network_block(network)
    t_sorted = sorted(block.t)
    outputs = {t: make_nn_output_expression(nn, [getattr(block, NORM_INPUT_NAME)[t, j] for j in range(problem.input_dim)]) for t in t_sorted}
    block.nn_output_constr = pyo.Constraint(t_sorted, block.nn_output_set, rule=lambda b, t, k: getattr(b, NORM_OUTPUT_NAME)[t, k] == outputs[t][k])
    m.obj = pyo.Objective(expr=0.0)
    _start_from_integration(block, ic, params, network)
    _add_layers(m, nn)
    # Each weight is a Var pinned to a mutable Param by one equality, so pounce can differentiate with respect to the Param.
    fixed = _weights(nn)
    m.weight_keys = pyo.Set(initialize=range(len(fixed)))
    m.weight = pyo.Param(m.weight_keys, mutable=True, initialize={k: float(v.value) for k, v in enumerate(fixed)})
    for v in fixed:
        v.unfix()
    m.weight_pin = pyo.Constraint(m.weight_keys, rule=lambda mo, k: fixed[k] == mo.weight[k])
    declare_sens_param(m.weight)
    weights = [m.weight[k] for k in m.weight_keys]
    solved = PounceInProcess().solve(m)
    ok = solved.result.solver.termination_condition == pyo.TerminationCondition.optimal
    if not ok:
        return None, solved.timing, found
    # Var(x) = (dx/dθ) Cov(θ) (dx/dθ)ᵀ at each collocation time.
    sd = {}
    for name, targets in (("prey", [block.x[t, 0] for t in t_sorted]), ("predator", [block.x[t, 1] for t in t_sorted]), ("z", [block.z[t, 0] for t in t_sorted])):
        j = _derivatives(targets, weights)
        sd[name] = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", j, network["cov"], j), 0.0)).tolist()
    found["sd"] = sd
    return extract_instance_data(problem, m), solved.timing, found


def simulate(options):
    """Step 3, the inference: the true model and the model with the trained network in place of z0, from each initial condition in turn."""
    o = json.loads(options)
    network = STATE["network"]
    params = STATE["params"]
    started = time.perf_counter()
    rows = []
    # The true model's and the trained model's x0, x1 and z0 at the trained
    # model's times, the true model read from its collocation polynomial,
    # stacked over the initial conditions that solve.
    stacked = {name: ([], [], []) for name in ("prey", "predator", "z")}
    for ic in o["ics"]:
        problem = LeslieGowerProblem(ics=np.array([ic], dtype=float), params=dict(params), t_span=(0.0, o["t_end"]), nfe=o["nfe"], ncp=NCP)
        truth = generate_data(problem=problem, noise_std=None, obs_every=1, seed=0, nlp_solver=PounceInProcess())
        hybrid, timing, found = _hybrid(ic, params, network, o["t_end"], o["nfe"])
        row = {"ic": list(ic), "timing": timing, **found, "truth": _trajectories(truth)[0] if truth is not None else None}
        if hybrid is not None:
            row["hybrid"] = _trajectories(hybrid)[0]
            row["hybrid"]["true_z"] = _true_z(params, row["hybrid"]["prey"], row["hybrid"]["predator"]).tolist()
            row["hybrid"]["sd"] = row.pop("sd")
            if truth is not None:
                for name, values, truths in (("prey", hybrid[0].nn_input[:, 0], truth[0].nn_input[:, 0]), ("predator", hybrid[0].nn_input[:, 1], truth[0].nn_input[:, 1]), ("z", hybrid[0].nn_output[:, 0], truth[0].nn_output[:, 0])):
                    stacked[name][0].extend(_collocation_value(truth[0].sampling_times, truths, t) for t in hybrid[0].sampling_times)
                    stacked[name][1].extend(values)
                    stacked[name][2].extend(row["hybrid"]["sd"][name])
        rows.append(row)
    metrics = {name: {**_fit(truths, values), "inside": _inside(truths, values, sds)} for name, (truths, values, sds) in stacked.items()} if stacked["prey"][0] else None
    return json.dumps({"ok": True, "ms": 1000 * (time.perf_counter() - started), "rows": rows, "metrics": metrics})
