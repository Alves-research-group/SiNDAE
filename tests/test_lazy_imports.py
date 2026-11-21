"""The expression-writing path imports and builds without JAX.

``import sindae`` and the modules ``build_simultaneous_model`` reaches import
JAX, Equinox, Optax and pynumero's grey-box interface only when a function
that calls them runs. Each test runs in a fresh interpreter, since this
session's ``sys.modules`` already has JAX from the other tests.
"""
import subprocess
import sys
import textwrap

DEFERRED = ["jax", "equinox", "optax", "pyomo.contrib.pynumero.interfaces.external_grey_box"]


def _run(code):
    done = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)], capture_output=True, text=True
    )
    assert done.returncode == 0, done.stderr
    return done.stdout.split()


def test_expression_path_imports_without_jax():
    loaded = _run(
        f"""
        import sys
        import sindae
        import sindae.algorithms.simultaneous.model_builder
        import sindae.algorithms.smoother
        import sindae.data_utils
        import sindae.example_problems
        import sindae.solvers
        print(*[m for m in {DEFERRED!r} if m in sys.modules])
        """
    )
    assert loaded == []


def test_numpy_network_with_named_activations_builds_nn_block():
    loaded = _run(
        f"""
        import sys
        from types import SimpleNamespace
        import numpy as np
        from sindae.algorithms.simultaneous.nn_block import eqx_mlp_to_nn_block

        rng = np.random.default_rng(0)
        sizes = [2, 4, 4, 1]
        layers = [
            SimpleNamespace(weight=rng.normal(size=(o, i)), bias=rng.normal(size=o))
            for i, o in zip(sizes, sizes[1:])
        ]
        net = SimpleNamespace(
            in_size=2, out_size=1, widths=[4, 4], activations=["tanh", "softplus"], layers=layers
        )
        block = eqx_mlp_to_nn_block(net)
        assert block.activations == ["tanh", "softplus"]
        for layer, built in zip(layers, block.layers):
            assert np.array_equal(built._w, layer.weight)
            assert np.array_equal(built._b, layer.bias)
        print(*[m for m in {DEFERRED!r} if m in sys.modules])
        """
    )
    assert loaded == []
