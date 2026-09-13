"""Build paper/images/vignette_grid.png (\\autoref{fig:fig2}).

Renders two code panels straight out of ``example.ipynb`` with Pygments, then
composes them with the two plot PNGs the executed notebook writes into
``paper/images/``.  Run from this directory *after* executing the notebook:

    jupyter nbconvert --to notebook --execute --inplace example.ipynb
    python make_vignette_grid.py
"""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import ImageFormatter

PAPER = pathlib.Path(__file__).resolve().parent
IMG = PAPER / "images"

nb = json.loads((PAPER / "example.ipynb").read_text())
code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]

# cell order: 0 imports, 1 problem class, 2 params/data, 3 fit, 4 fit-plot,
#             5 law-plot, 6 predict, 7 predict-plot
problem_src = "".join(code_cells[1]["source"]).rstrip()
fit_src = "".join(code_cells[3]["source"]).rstrip()
predict_src = "".join(code_cells[6]["source"]).rstrip()

# The predict cell's ground-truth line is example bookkeeping, not workflow.
predict_src = "\n".join(
    ln for ln in predict_src.splitlines() if "val_truth" not in ln
).rstrip()

fit_panel = fit_src + "\n\n" + predict_src


def render(src, path, font_size=30):
    fmt = ImageFormatter(
        font_name="Menlo",
        font_size=font_size,
        line_numbers=False,
        style="default",
        image_pad=18,
        line_pad=4,
    )
    path.write_bytes(highlight(src, PythonLexer(), fmt))
    return path


render(problem_src, IMG / "vignette_code_problem.png")
render(fit_panel, IMG / "vignette_code_fit.png")

panels = {
    "a": mpimg.imread(IMG / "two_tank_diagram.png"),      # system schematic
    "b": mpimg.imread(IMG / "vignette_code_problem.png"),
    "c": mpimg.imread(IMG / "vignette_code_fit.png"),
    "d": mpimg.imread(IMG / "vignette_law.png"),
    "e": mpimg.imread(IMG / "vignette_predict.png"),
}
asp = {k: v.shape[1] / v.shape[0] for k, v in panels.items()}
for k, v in panels.items():
    print(k, v.shape, f"aspect {asp[k]:.2f}")

# Every cell is sized to its panel's measured aspect, so nothing letterboxes.
# Layout, in units of total figure width (= 1):
#   top block   left column  = (a) schematic over (c) fit/export/predict code
#               right column = (b) problem definition, spanning both rows
#   bottom row  = (d) learned law beside (e) prediction
w_left = 1.0 / (1.0 + asp["b"] * (1.0 / asp["a"] + 1.0 / asp["c"]))
w_right = 1.0 - w_left
h_top = w_left * (1.0 / asp["a"] + 1.0 / asp["c"])

w_law = asp["d"] / (asp["d"] + asp["e"])
h_bot = 1.0 / (asp["d"] + asp["e"])

FIG_W = 14.0
GAP = 0.20  # gap between the top block and the bottom row, inches
FIG_H = FIG_W * (h_top + h_bot) + GAP

# Two gridspecs placed by hand rather than fig.subfigures: subfigures split the
# whole canvas by height_ratios, which folds GAP back into the blocks and
# reopens the dead band under the left column.
fig = plt.figure(figsize=(FIG_W, FIG_H))
gs_top = fig.add_gridspec(2, 2,
                          width_ratios=[w_left, w_right],
                          height_ratios=[1.0 / asp["a"], 1.0 / asp["c"]],
                          left=0.0, right=1.0,
                          bottom=(FIG_W * h_bot + GAP) / FIG_H, top=1.0,
                          wspace=0.03, hspace=0.06)
gs_bot = fig.add_gridspec(1, 2, width_ratios=[w_law, 1.0 - w_law],
                          left=0.0, right=1.0,
                          bottom=0.0, top=FIG_W * h_bot / FIG_H,
                          wspace=0.10)

axes = {
    "a": fig.add_subplot(gs_top[0, 0]),
    "b": fig.add_subplot(gs_top[:, 1]),
    "c": fig.add_subplot(gs_top[1, 0]),
    "d": fig.add_subplot(gs_bot[0, 0]),
    "e": fig.add_subplot(gs_bot[0, 1]),
}

for key in "abcde":
    ax = axes[key]
    ax.imshow(panels[key])
    ax.set_axis_off()
    ax.set_anchor("N" if key in "abc" else "C")
    ax.text(-0.01, 1.0, f"({key})", transform=ax.transAxes,
            ha="right", va="top", fontsize=17, fontweight="bold")

fig.savefig(IMG / "vignette_grid.png", dpi=200, bbox_inches="tight",
            facecolor="white")
print("wrote", IMG / "vignette_grid.png")
