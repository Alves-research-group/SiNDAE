---
title: 'SiNDAE: The Simultaneous Neural Differential-Algebraic systems of equations training Python package'
tags:
  - Python
  - Nonlinear Programming
  - Neural DAEs
  - Dynamic Optimization
  - Process systems modeling
  - Hybrid modeling
  - Differentiable Optimization
authors:
  - name: Laurens Lueg
    affiliation: 1
  - name: Nicolas Smits
    affiliation: 1
  - name: Victor Alves
    affiliation: 1
  - name: John R. Kitchin
    affiliation: 1
  - name: Carl D. Laird
    affiliation: 1
  - name: Lorenz T. Biegler
    affiliation: 1
  
affiliations:
 - name: Department of Chemical Engineering, Carnegie Mellon University, Pittsburgh, Pennsylvania, USA
   index: 1

date: 1 September 2026
bibliography: references.bib
---

# Summary

## (1) The Problem
When modeling dynamical systems, first-principles mechanistic, constitutive, or approximate equations are often incomplete or insufficient to capture the true underlying complexity of the phenomena at play. Depending on the system, the modeling shortfall could manifest itself as for example, a reaction rate, growth term, or transfer coefficient which could depend on measured state variables via a relationship which is not explicitly modeled but is present in experimental data. Additionally, practitioners are often aware of the factors which might be influencing experimental deviations from a model but may not have the time, capacity, or deep modeling knowledge to formulate, solve, and verify new mechanistic equations. Hybrid modeling solve these dilemmas by leveraging existing mechanistic knowledge while using data-driven surrogates to capture any un-modeled relationships present in the data. In this way, Hybrid models have a two-fold advantage over data-driven or purely-mechanistic approaches. Unlike data-driven models, hybrid models can encode universal truths such as mass, energy, or momentum balances into solution routines, allowing mappings learned from data to remain physically consistent. Simultaneously, unlike purely-mechanistic models, hybrid models are able to use valuable experimental data to inform and improve the solution of dynamical systems. Moreover, the mechanistic-data-driven trade-off can be tuned to achieve the best performance based on the modelling goal, whether that be simulation, model-based-design of experiments, embedded surrogate optimization etc [@mahanty2023hybrid]. The benefits of hybrid approaches are well established: a recent review of 270 publications spanning the early 1990s to 2024 reports systematic gains in predictive accuracy and extrapolation capability across bioprocess applications [@herreraruiz2025hybrid], and the approach dates to the first hybrid neural network process models of the early 1990s [@psichogios1992hybrid;@thompson1994modeling]. Most available tooling, however, targets hybrid Ordinary Differential Equation (ODE) systems, while comparatively little addresses hybrid Differential-Algebraic Equation (DAE) systems. SiNDAE is an open source package which extends hybrid modeling functionality to DAEs.


## (2) Background
In the field of process systems modeling in science and engineering, paradigms combining data-driven components with purely mechanistic models have gained traction as promising hybrid improvements on traditional methods. The milestone approaches published in this area of research include: Physics Informed Neural Networks (PINNs) [@Raissi2019PINNs], Neural Ordinary Differential Equations (NeurODEs) [@chen2018neural], and Universal Differential Equations (UDEs) [@rackauckas2021universal]. Said hybrid methods have been applied to solving ODEs domains ranging from bioprocesses [@Bangi2022PINNs;@Narayanan2022Hybrid] to power systems [@Xiao2023PowerSystems] to wastewater treatment [@Huang2025Wastewater] [@lueg2025simultaneous]. Recently, work in the field has extended to hybridizing Differential Algebraic Systems of Equations (DAEs) in order to allow data-driven components to obey explicit algebraic constraints. While works focusing on the reconciliation of learned models with constraints [@Mukherjee2025MEB] or PINNs learning the solution of DAE systems with physics informed loss functions, fewer works focus on approximating unknown components of DAE systems. The hybrid DAE training approaches that do exist compute derivatives of the loss with respect to the network parameters through a separate solution step, using integration and adjoint sensitivities [@kim2021stiff], stabilization or projection onto the constraint manifold [@white2023stabilized], operator splitting between the differential and algebraic subsystems [@koch2024operator], or penalty terms on the constraint residual [@moya2022daepinn]. These are sequential strategies, and in most cases the algebraic constraints are satisfied only approximately: stabilization keeps physical invariants approximately satisfied during simulation [@white2023stabilized], and penalty formulations enforce the DAE as approximate hard constraints whose accuracy depends on hyperparameters that, chosen poorly, leave the optimization ill-conditioned or slow to converge [@moya2022daepinn]. 

## (3) How SiNDAE fills the gap in the field
The difficulty is intrinsic rather than incidental. DAEs present a form of infinite stiffness, which produces gradient pathologies and ill-conditioned optimization problems that cause gradient-descent-based training to fail outright [@moya2022daepinn], and the same effect is documented for stiff ODEs, where adjoint sensitivities computed by the optimise-then-discretise route break down entirely [@kim2021stiff], with higher-index systems compounding this and other issues [@white2023stabilized]. The consequence for tooling is that the mature neural differential equation frameworks are ODE frameworks. `torchdiffeq` [@chen2018neural], `DiffEqFlux.jl` [@rackauckas2019diffeqflux], and `diffrax` [@kidger2022neuralde] all train by integrating the system forward and backpropagating through, or around, the solver, with none providing a mechanism for holding hard algebraic constraints on the training problem itself. Work that does target DAEs largely learns the solution trajectory of a fully specified system [@moya2022daepinn] rather than an unknown component of one.

Lueg et al. propose two novel training strategies to tackle rigorous DAE constraints enforcement, solving higher-index systems, and scalability to many trajectories. The simultaneous approach employs a strategy similar to simultaneous approaches for solving purely-mechanistic DAEs by discretizing the domain using orthogonal collocation before solving the system using a capable nonlinear solver such as IPOPT [@wachter2006ipopt]. Said method effectively achieves constraint enforcement and higher-index system solutions, scaling to moderately sized systems. Conversely, the decomposition approach adapts training to a bi-level optimization problem, where neural network weights are updated via a sensitivity-based stochastic gradient descent outer loop, while interior iterations solve subproblems parallelized across trajectories with the neural network parameters frozen. Although the decomposition approach solves the DAE and updates the weights in alternation, it retains rigorous constraint enforcement and higher-index performance while scaling to large numbers of trajectories.

## (4) What SiNDAE is 
SiNDAE implements the simultaneous and decomposition approaches from *A Simultaneous Approach for Training Neural DAEs* by Lueg et al. [@lueg2025simultaneous], bridging the tooling and functionality gap by allowing components of DAEs to be learned subject to algebraic constraints in a scalable fashion. The package leverages `Pyomo` and `Pyomo.DAE` to build the NLPs solved in both approaches, ensuing both rigorous constraint enforcement and sufficient modeling flexibility for diverse applications. SiNDAE's API design is indented to be user-friendly, mirroring widespread machine learning packagess architecture, namely scikit-learn's [@scikit-learn] and SparkML [@SparkMLlib] estimator oriented design. Similarly, its installation intended to be hassle-free by alleviating the need for external binaries for the linear and nonlinear solvers required, thus streamlining adoption. Interoperability into complementary machine learning, modeling, and optimization ecosystems is facilitated through SiNDAE's export formats, enabiling the dissemination of accurate hybrid models for use in downstream use-cases. Together, SiNDAE's functionality and design is intended to facilitate the training of hybrid DAE systems in as user-friendly format, with auxiliary integrations kept in mind.

# Statement of need

Why is this package needed in the field? How would this package benefit the community? What frameworks in this python package built on? Dependency graph?


Namely, SiNDAE makes use of a similar estimator object philosophy adapted for hybrid DAEs through the `HybridDAE` and `ProblemDefinition` entry points, reducing the learning curve for practitioners already familiar with similarly designed frameworks. In the same spirit of usability, unlike other Python packages making use of state of the art linear/nonlinear solvers (e.g. MA27 or IPOPT), SiNDAE features an entirely licensed-binary-free default installation by using the FERAL [@kitchin2026feral] and POUNCE [@kitchin2026pounce] solvers.


![Dependency graph generated with [pydeps](https://github.com/thebjorn/pydeps/) showing the packages used in ``SiNDAE``.\label{fig:fig1}](./images/sindae_pydeps.png)

lack of code sharing in the community [@mahanty2023hybrid]

CLAUDE integration

# Vignette

Point to examples gallery directory in the docs.

Describe the example.

Code snippets of solve.

Mention of the claude one shot implementation.

# Availability

``SiNDAE`` is available on [PyPI](INSERT PYPI LINK) and has its source code hosted on [GitHub](https://alves-research-group.github.io/SiNDAE/). The documentation contains
thorough [descriptions of the API and functionality](INSERT DOCS PAGE LINK), as well as [theoretical grounding](INSERT DOCS PAGE LINK) for both of the solution methods available, [instructions](INSERT DOCS LINK) on how to configure a hybrid DAE system, and a [gallery of examples](INSERT DOCS EXAMPLES LINK). 

The idea is to supply both proper documentation to
the users in the open-source software community as well as to give the users the necessary amount of theory allowing them to employ process operability principles in their specific application.
 

# Acknowledgements

# References