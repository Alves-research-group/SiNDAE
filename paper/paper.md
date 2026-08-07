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
NOTE: Will delete subheadings later.

## (1) The Problem
When modeling dynamical systems, first-principles mechanistic, constitutive, or approximate equations are often incomplete or insufficient to capture the true underlying complexity of the phenomena at play. Depending on the system, the modeling shortfall could manifest itself as for example, a reaction rate, growth term, or transfer coefficient which could depend on measured state variables via a relationship which present yet not explicitly modeled. While domain experts may recognize these missing dynamics, formulating and validating new governing equations is often intractable. Hybrid modeling resolves these dilemmas by leveraging existing mechanistic knowledge while embedding data-driven surrogates to improve fitting and generalization performance. In this way, hybrid models have a two-fold advantage over data-driven or purely-mechanistic approaches. Unlike data-driven models, hybrid models can enforce physical invariants such as mass, energy, or momentum balances into solution routines, allowing mappings learned from data to remain physically consistent. Simultaneously, unlike purely-mechanistic models, hybrid models are able to use valuable experimental data to inform and improve the solution of dynamical systems. The mechanistic-data-driven trade-off can be tuned to achieve the best performance based on the modelling goal, whether that be simulation, model-based-design of experiments, embedded surrogate optimization etc [@mahanty2023hybrid]. 

While the predictive and extrapolative benefits of hybrid modeling are well-documented across decades of literature [@psichogios1992hybrid;@thompson1994modeling;@herreraruiz2025hybrid;@Narayanan2022Hybrid], existing open-source software targets Ordinary Differential Equation (ODE) systems almost exclusively. Comparatively little tooling addresses hybrid Differential-Algebraic Equation (DAE) systems, where algebraic constraints must hold alongside differential dynamics. SiNDAE fills this software gap by extending scalable hybrid modeling functionality to DAE systems.

## (2) Background 
In the field of process systems modeling in science and engineering, paradigms combining data-driven components with purely mechanistic models have gained traction as promising hybrid improvements on traditional methods. The milestone approaches published in this area of research include: Physics Informed Neural Networks (PINNs) [@Raissi2019PINNs], Neural Ordinary Differential Equations (NeurODEs) [@chen2018neural], and Universal Differential Equations (UDEs) [@rackauckas2021universal]. Said hybrid methods have been applied to solving ODEs domains ranging from bioprocesses [@Bangi2022PINNs;@Narayanan2022Hybrid] to power systems [@Xiao2023PowerSystems] to wastewater treatment [@Huang2025Wastewater]. However, mature scientific machine learning frameworks such as torchdiffeq [@chen2018neural], DiffEqFlux.jl [@rackauckas2019diffeqflux], and diffrax [@kidger2022neuralde] rely on integrate-and-backpropagate mechanisms designed for ODEs. They lack native mechanisms for enforcing explicit, hard algebraic constraints during model training. 

Recently, work in the field has extended to hybridizing Differential Algebraic Systems of Equations (DAEs) to obey explicit algebraic constraints has relied on sequential training or penalty loss functions. These methods use techniques such as integration and adjoint sensitivities [@kim2021stiff], stabilization or projection onto the constraint manifold [@white2023stabilized], operator splitting between the differential and algebraic subsystems [@koch2024operator], or penalty terms on the constraint residual [@moya2022daepinn]. Consequently, algebraic constraints are enforced only softy or approximately. Stabilization methods keep physical invariants only approximately satisfied during simulation [@white2023stabilized], and penalty formulations enforce the DAE as approximate hard constraints whose accuracy depends on hyperparameters that, chosen poorly, leave the optimization ill-conditioned or slow to converge [@moya2022daepinn]. Furthermore, most existing DAE-focused PINN research aims to learn trajectory solutions of fully specified systems [@moya2022daepinn], rather than discovering unknown functional components within constrained systems.

To overcome theses issues, Lueg et al. propose two novel training strategies to tackle rigorous DAE constraints enforcement, solving higher-index systems, and scalability to many trajectories. The simultaneous approach employs a strategy similar to simultaneous approaches for solving purely-mechanistic DAEs by discretizing the domain using orthogonal collocation before solving the system using a capable nonlinear solver such as IPOPT [@wachter2006ipopt]. Said method effectively achieves constraint enforcement and higher-index system solutions, scaling to moderately sized systems. Conversely, the decomposition approach adapts training to a bi-level optimization problem, where neural network weights are updated via a sensitivity-based stochastic gradient descent outer loop, while interior iterations solve subproblems parallelized across trajectories with the neural network parameters frozen. Although the decomposition approach solves the DAE and updates the weights in alternation, it retains rigorous constraint enforcement and higher-index performance while scaling to large numbers of trajectories.

## (3) What SiNDAE is 
SiNDAE implements the simultaneous and decomposition approaches from *A Simultaneous Approach for Training Neural DAEs* by Lueg et al. [@lueg2025simultaneous], bridging the tooling and functionality gap by allowing components of DAEs to be learned subject to algebraic constraints in a scalable fashion. The package leverages `Pyomo` and `Pyomo.DAE` to build the NLPs solved in both approaches, ensuing both rigorous constraint enforcement and sufficient modeling flexibility for diverse applications. SiNDAE's API design is indented to be user-friendly, mirroring widespread machine learning packagess architecture, namely scikit-learn's [@scikit-learn] and SparkML [@SparkMLlib] estimator oriented design. Similarly, its installation intended to be hassle-free by alleviating the need for external binaries for the linear and nonlinear solvers required, thus streamlining adoption. Interoperability into complementary machine learning, modeling, and optimization ecosystems is facilitated through SiNDAE's export formats, enabiling the dissemination of accurate hybrid models for use in downstream use-cases. Together, SiNDAE's functionality and design is intended to facilitate the training of hybrid DAE systems in as user-friendly format, with auxiliary integrations kept in mind.

# Statement of need

## (5) The Audience & The Scientific Gap:
SiNDAE addresses a critical gap in scientific machine learning by providing a dedicated framework for hybrid Differential-Algebraic Equation (DAE) modeling. The package targets researchers and practitioners in process systems engineering, chemical engineering, and applied physics who hold mechanistic DAE models with uncharacterized or uncertain terms (e.g., complex reaction kinetics or mass transfer coefficients) and possess noisy, incomplete state measurements. While existing hybrid differential equation packages cater almost exclusively to Ordinary Differential Equations (ODEs), SiNDAE allows users to learn unknown functional components directly within index-1 and higher-index DAE systems. By enforcing governing algebraic equations as hard constraints during both training and inference, SiNDAE ensures that unobserved state variables remain physically consistent, enabling robust extrapolation to unmeasured operating regimes—a defining advantage of hybrid modeling [@vonstosch2014hybrid; @herreraruiz2025hybrid].

## (6) Usability & Open Access
A major barrier to the adoption and open reproduction of hybrid modeling workflows is the lack of open-source software and the reliance on proprietary or complex solver installation stacks [@mahanty2023hybrid]. Traditional dynamic optimization workflows often depend on solvers like IPOPT [@wachter2006ipopt] paired with linear solvers (e.g., HSL's MA27) that require specialized licenses and compiled binaries, hindering code sharing across communal repositories. SiNDAE overcomes this hurdle by featuring a completely license-binary-free default installation powered by the POUNCE [@kitchin2026pounce] nonlinear solver and the FERAL [@kitchin2026feral] linear solver, both distributed directly via standard Python wheels. Practitioners can execute the entire pipeline—from parameter pre-training and simultaneous/decomposition training to model inference—out of the box, while retaining optional support for traditional cyipopt backends. Furthermore, SiNDAE adopts an estimator-oriented API architecture modeled after scikit-learn (HybridDAE and ProblemDefinition), reducing the learning curve for ML practitioners. Trained models can be exported to Equinox, ONNX, JSON, or OMLT NetworkDefinition objects [@ceccon2022omlt], allowing identified surrogates to be embedded seamlessly into downstream optimization and control workflows. Additionally, to enable expedited hybrid modeling workflows using modern AI coding agents (primarily Claude Code), a directory called `sindae-skills` containing an instruction file and several skill files has been included. When configured for a coding agent working on a SiNDAE project, the instruction file guides the user interaction and modelling process through latex equation rendering and transparency rules, while skill files provide design decision and implementation best practices.

## (7) Ecosystem & Dependencies
SiNDAE is built upon established scientific Python infrastructure. Symbolic model construction and Lagrange–Radau collocation on finite elements are managed via Pyomo and Pyomo.DAE [@bynum2021pyomo;@nicholson2018pyomodae], while Pyomo's PyNumero interface exposes the underlying NLP structures and KKT systems for implicit differentiation. Neural network architectures, exact higher-order automatic differentiation, and outer-loop optimization updates are powered by JAX [@jax2018github], Equinox [@kidger2021equinox], and Optax [@deepmind2020jax]. Fundamental data manipulation, scientific routines, and visualization are supported by NumPy, SciPy, and Matplotlib. Parallelization of the decomposition strategy is enabled through the optional use of MPI for Python [@dalcin2005mpi]. Exporting and printing functonality are handled by the jax2onnx [@jax2onnx] and tabulate [@tabulate] packages respectively. \autoref{fig:fig1} illustrates the resulting dependency architecture.

![Dependency graph generated with [pydeps](https://github.com/thebjorn/pydeps/) showing the packages used in ``SiNDAE``.\label{fig:fig1}](./images/sindae_pydeps.png)

# Vignette

To illustrate `SiNDAE`'s core capabilities and user workflow, consider a classic process engineering system: a two-tank liquid cascade operating as an index-1 Differential-Algebraic Equation (DAE). In this system, liquid levels $h_1(t)$ and $h_2(t)$ evolve under a constant feed rate $F_{in}$ and inter-tank flows $q_1(t)$ and $q_2(t)$:

$$
\begin{align}
A_1 \frac{dh_1}{dt} = F_{in} - q_1, \qquad A_2 \frac{dh_2}{dt} = q_1 - q_2, \qquad q_2 = c_2\sqrt{h_2}
\end{align}
$$

In practical scenarios, physical relationships are often only partially known. Here, the outlet relation of the second tank ($q_2$) is explicitly governed by a known valve equation, while the inter-tank discharge law ($q_1$) is unknown and must be learned from noisy, sparse trajectory data of the liquid levels.

The core philosophy of `SiNDAE` is to eliminate the mathematical and software engineering overhead traditional in dynamic optimization. Rather than requiring users to manually formulate orthogonal collocation matrices, derive sensitivity relations, or manage complex C++/Fortran solver dependencies, `SiNDAE` encapsulates the entire hybrid training and inference pipeline into a high-level, estimator-oriented architecture. 

To solve this problem using `SiNDAE`, a user only needs to perform three concise steps:

1. **Define the Physics:** Subclass `ProblemDefinition` to specify the mechanistic DAE in `Pyomo` format and designate which variables enter and exit the neural surrogate (\autoref{fig:fig2}a).
2. **Configure and Fit:** Assemble the multi-stage training pipeline by pairing the problem with stage configuration objects. Calling `fit()` automatically handles data smoothing, supervised network pre-training, and the simultaneous non-linear program (NLP) optimization solve (\autoref{fig:fig2}b). Users can seamlessly swap between the simultaneous and decomposition algorithms or customize neural architectures via `SiNDAE`'s grey-box interface.
3. **Predict and Export:** Call `predict()` on the fitted estimator to evaluate new initial conditions or operating regimes (\autoref{fig:fig2}b). Because `predict()` re-embeds the learned surrogate into the full mechanistic DAE solver, all predicted trajectories satisfy hard algebraic constraints by construction.

As shown in \autoref{fig:fig2}c, although the neural network $f_{NN}(h_1, h_2; \theta)$ receives no prior structural information, its learned response collapses onto the true physics ($c_1\sqrt{h_1}$). Furthermore, predictions under unseen initial charges track reference trajectories exactly (\autoref{fig:fig2}d). Once identified, the trained hybrid model can be directly exported to ONNX, Equinox, or `OMLT` formats for use in downstream surrogate optimization and control tasks.

![A complete ``SiNDAE`` workflow on a two-tank cascade DAE. (a) system definition via a ``ProblemDefinition`` subclass; (b) estimator assembly, multi-stage ``fit``, and constrained ``predict`` execution; (c) recovery of the un-modeled discharge law against the physical truth; (d) physically consistent prediction on holdout initial conditions.\label{fig:fig2}](./images/vignette_grid.png)


# Availability

``SiNDAE`` is available on [PyPI](https://pypi.org/project/sindae/) and has its source code hosted on [GitHub](https://alves-research-group.github.io/SiNDAE/). The documentation contains
thorough [descriptions of the API and functionality](https://alves-research-group.github.io/SiNDAE/index-1/), as well as [theoretical](https://alves-research-group.github.io/SiNDAE/hybrid-dae-overview/) grounding for both the [simultaneous](https://alves-research-group.github.io/SiNDAE/simultaneous-solver/) and the [decomposition](https://alves-research-group.github.io/SiNDAE/decomposition-solver/) methods available, [instructions](https://alves-research-group.github.io/SiNDAE/hybrid-dae/) on how to configure a hybrid DAE system, as well as a [gallery of examples](https://alves-research-group.github.io/SiNDAE/index-2/). 

The authors aim to supply both proper documentation to the users in the open-source software community and provide sufficient grounding in theory to enable them to design, implement, and distribute hybrid DAE models for their use-case.
 

# Acknowledgements

# References