"""SiNDAE's public names.

Each name imports its module on first use (PEP 562), so ``import sindae`` and
the modules the expression-writing path uses (``problem``, ``data_utils``,
``solvers``, ``algorithms.smoother`` and ``algorithms.simultaneous.model_builder``)
import without JAX, Equinox, Optax or pynumero's grey-box interface. Those are
imported by the modules that call them, when they are first used.
"""

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version

_EXPORTS = {
    "SimpleMLP": "sindae.nn_utils",
    "flatten_fn": "sindae.nn_utils",
    "make_unflatten_fn": "sindae.nn_utils",
    "InstanceData": "sindae.data_utils",
    "TrajectoryData": "sindae.data_utils",
    "NormStats": "sindae.data_utils",
    "extract_instance_data": "sindae.data_utils",
    "generate_data": "sindae.data_utils",
    "ProblemDefinition": "sindae.problem",
    "HybridDAE": "sindae.hybrid_dae",
    "make_nlp_solver": "sindae.solvers",
    "make_linear_solver": "sindae.solvers",
    "NLPSolver": "sindae.solvers",
    "SolverConfig": "sindae.solvers",
    "build_smoother_model": "sindae.algorithms.smoother",
    "solve_smoother": "sindae.algorithms.smoother",
    "SmootherConfig": "sindae.algorithms.smoother",
    "PretrainConfig": "sindae.algorithms.pretrain",
    "pretrain_mlp": "sindae.algorithms.pretrain",
    "DecompConfig": "sindae.algorithms.decomp.train",
    "train_decomp": "sindae.algorithms.decomp.train",
    "build_decomp_model": "sindae.algorithms.decomp.model_builder",
    "build_simultaneous_model": "sindae.algorithms.simultaneous.model_builder",
    "build_simultaneous_model_gbm": "sindae.algorithms.simultaneous.model_builder",
    "extract_mlp": "sindae.algorithms.simultaneous.model_builder",
    "SimultaneousConfig": "sindae.algorithms.simultaneous.train",
    "solve_simultaneous": "sindae.algorithms.simultaneous.train",
    "make_inference_model": "sindae.algorithms.inference",
    "solve_inference": "sindae.algorithms.inference",
    "plot_instance_data": "sindae.plot_utils",
    "plot_training_history": "sindae.plot_utils",
    "FourTankProblem": "sindae.example_problems",
    "LeslieGowerProblem": "sindae.example_problems",
    "FedBatchBioreactorProblem": "sindae.example_problems",
}

__all__ = [*_EXPORTS, "__version__"]

try:
    __version__ = version("sindae")
except PackageNotFoundError:
    # Plain source checkout without an installed distribution.
    __version__ = "0.0.0+unknown"


def __getattr__(name):
    """Import the module that defines ``name``, or the submodule ``name``."""
    if name in _EXPORTS:
        value = getattr(import_module(_EXPORTS[name]), name)
    else:
        try:
            value = import_module(f"{__name__}.{name}")
        except ModuleNotFoundError as error:
            if error.name != f"{__name__}.{name}":
                raise
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    globals()[name] = value
    return value


def __dir__():
    return sorted({*globals(), *_EXPORTS})
