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
from matplotlib.gridspec import GridSpec
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
    "a": mpimg.imread(IMG / "vignette_code_problem.png"),
    "b": mpimg.imread(IMG / "vignette_code_fit.png"),
    "c": mpimg.imread(IMG / "vignette_law.png"),
    "d": mpimg.imread(IMG / "vignette_predict.png"),
}

# Column/row ratios chosen so each cell matches its panel's natural aspect,
# leaving no letterboxing gaps.
fig = plt.figure(figsize=(14.0, 12.2))
gs = GridSpec(3, 2, figure=fig,
              width_ratios=[1.35, 1.0],
              height_ratios=[0.68, 0.79, 0.58],
              hspace=0.05, wspace=0.03)

axes = {
    "a": fig.add_subplot(gs[0:2, 0]),
    "b": fig.add_subplot(gs[0, 1]),
    "c": fig.add_subplot(gs[1, 1]),
    "d": fig.add_subplot(gs[2, :]),
}

for key in "abcd":
    ax = axes[key]
    ax.imshow(panels[key])
    ax.set_axis_off()
    ax.set_anchor("N" if key in "ab" else "C")
    ax.text(-0.01, 1.0, f"({key})", transform=ax.transAxes,
            ha="right", va="top", fontsize=17, fontweight="bold")

fig.savefig(IMG / "vignette_grid.png", dpi=200, bbox_inches="tight",
            facecolor="white")
print("wrote", IMG / "vignette_grid.png")
