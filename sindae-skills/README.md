# SiNDAE modeling skills for Claude

This folder is a drop-in knowledge bundle that turns Claude into a hybrid modeling
assistant for SiNDAE. This folder contains only md files.

The bundle is two things:

- `CLAUDE.md`, the operating instructions: the role, the mandatory workflow, the
  human-in-the-loop equation verification, and honesty rules.
- `skills/`, the reference documents Claude consults before it makes a modeling
  suggestion.

## Prerequisites

Claude needs to be able to run SiNDAE, not just read about it. Install the package
into the environment Claude will execute code in before you start a session:

```bash
pip install -e ".[full,test]"     # from a clone of this repository
pyomo build-extensions            # only if you will use grey-box or decomposition solves
```

## Setup

Pick the option that matches how you use Claude. 

### Option A: Claude Code, in your own modeling project

Claude Code reads `CLAUDE.md` from the working directory at the start of every
session. Copy the bundle so that `CLAUDE.md` lands at the root of the project where
your data and your model code are saved:

```bash
mkdir my-bioreactor-model
cp /path/to/SiNDAE/sindae-skills/CLAUDE.md /my-bioreactor-model/
cp /path/to/SiNDAE/sindae-skills/skills    /my-bioreactor-model/.claude/skills/
cd /my-bioreactor-model
claude
```

or move it manually.

The resulting layout:

```
my-bioreactor-model/
  CLAUDE.md             <- loaded automatically at session start
    .claude/
      skills/           <- read on demand, as CLAUDE.md directs
  data/                 <- your measurements
```

If you already have a `CLAUDE.md` in that project, do not overwrite it. Append the
contents of this one, or keep yours and add a line pointing at the bundle:

Verify it took effect by opening Claude in the directory of you project and running the command in the Claude code:
```python
/memory
```
this should open the CLAUDE.md file we just moved into your directory. 

### Option B: Claude Desktop or claude.ai

1. Create a Project.
2. Paste the contents of `CLAUDE.md` into the project's custom instructions.
3. Upload the files in `skills/` as project knowledge.

## What a session looks like

The workflow in `CLAUDE.md`:

1. **Model.** Claude asks for the states, known dynamics, parameters, the unknown
   term, what is measured, and the time span. It is instructed to ask rather than
   assume.
2. **Draft.** The governing equations are written out, with the network outputs
   marked. No code yet.
3. **Confirm.** Claude renders the full model for the user and stops. You approve or
   correct the equations before anything runs.
4. **Build.** The `ProblemDefinition` subclass is written.
5. **Validate.** Shape and dimensional checks, every state derivative constrained,
   the network output actually used.
6. **Fit and report.** `HybridDAE.fit`, then per-state metrics and plots, with
   `termination` reported.
7. **Predict.** `HybridDAE.predict` for new operating conditions, if you want them.


## File Breakdown

| File | Load when |
|------|-----------|
| `skills/sindae-hybridization-strategy.md` | The user is still deciding *what* to learn: degree of hybridization, the accuracy vs data-cost trade-off, extrapolation, interpretability. |
| `skills/sindae-formulation.md` | Writing the `ProblemDefinition` subclass: the Pyomo and collocation cookbook, attaching measurements, and the self-check to run before fitting. |
| `skills/sindae-method-selection.md` | Configuring a fit or a prediction: `method`, the solver stack, the config dataclasses, `unfix_io`, `slack_coef`. |
| `skills/sindae-worked-examples.md` | The new problem resembles the four-tank DAE, Leslie-Gower ODE, or fed-batch bioreactor. |
| `skills/sindae-failure-playbook.md` | A solve does not reach `optimal`, crashes, or diverges. |
