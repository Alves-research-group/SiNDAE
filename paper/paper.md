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

What is the problem that this package aims to tackle? Historical context? Other packages? What can't they do that this one does?

- Training neural DAEs is more challenging than nueral ODEs, specifically for higher index systems.

- Typical training approaches use methods including integration, projection, operator splitting, or penalty terms, each of which are sequential.

- This package encapsulates the simultaneous and decomposition methods described in [@lueg2025simultaneous] in a user-friendly package with a binary free install. Built using Pyomo, PyNumero, NLP solvers ... with the aim of enabling the training of constrained DAE systems depending on problem type.

## The Problem
When modeling dynamical systems, first-principles mechanistic, constitutive, and approximate equations are often incomplete or insufficient to capture the true underlying complexity of the phenomena at play. Depending on the system, the modeling shortfall could manifest as a reaction rate, a growth term, or a transfer coefficient which depend non-trivial

## Background
In the field of process systems modeling in science and engineering, paradigms combining data-driven components with purely mechanistic models have gained traction as promising *hybrid* improvements on traditional methods. The milestone approaches published in this area of research include: Physics Informed Neural Networks (PINNs) [@Raissi2019PINNs], Neurual Ordinary Differential Equations (NeurODEs) [@chen2018neural], and Universal Differential Equations (UDEs) [@rackauckas2021universal]. Said hybrid methods have been applied to solving ODEs domains ranging from bioprocesses [@Bangi2022PINNs] [@Narayanan2022Hybrid] to power systems [@Xiao2023PowerSystems] to wastewater treatment [@Huang2025Wastewater] [@lueg2025simultaneous]. 

Recently, work in the field has extended to hybridizing Differential Algebraic Systems of Equations (DAEs) in order to allow data-driven components to obey explicit algebraic constraints. While works focusing on the reconciliation of learned models with constraints [@Mukherjee2025MEB] or PINNs learning the solution of DAE systems with physics informed loss functions, fewer works focus on approximating unknown components of DAE systems. The hybrid DAE training approaches which exist utilize derivatives of the loss function with respect to the neural network parameters which are computed through operations including integration [ref], projection [ref], operator splitting [ref], or constraint penalty terms [ref] in a so-called *sequential* manner. 

Lueg et al propose two novel training strategies to tackle rigorous DAE constraints enforcement, solving higher-index systems, and scalability to many trajectories. The simultaneous approach employs a strategy similar to simultaneous approaches for solving purely-mechanistic DAEs by discretizing the domain using orthogonal collocation before solving the system using a capable nonlinear solver such as IPOPT [@wachter2006ipopt]. Said method effectively achieves constraint enforcement and higher-index system solutions, performing up to moderately sized systems. Conversely, the decomposition approach adapts training to a bi-level optimization problem, where neural network weights are updated via a sensitivity-based stochastic gradient descent outer loop, while interior iterations solve subproblems parallelized across trajectories with the neural network parameters frozen. While sequential, the latter approach retains constraint-enforcement and higher-index problem performance, but also scales effectively to large numbers of trajectories.

## What SiNDAE is 
SiNDAE implements the simultaneous and decomposition approaches from *A Simultaneous Approach for Training Neural DAEs*  [@lueg2025simultaneous], bridging the functionality gap by allowing components of DAEs to be learned subject to algebraic constraints in a scalable fashion. The package leverages `Pyomo` and `Pyomo.DAE` to build the NLPs solved in both approaches, ensuing both rigorous constraint enforcement and sufficient modeling flexibility for diverse applications.

SiNDAE's API design is indented to be user-friendly, mirroring widespread machine learning packages including scikit-learn's [@scikit-learn] and SparkML [@SparkMLlib]. Similarly, its installation intended to be hassle-free by alleviating the need for external binaries for the linear and nonlinear solvers required. 

# Statement of need

Why is this package needed in the field? How would this package benefit the community? What frameworks in this python package built on? Dependency graph?


Namely, SiNDAE makes use of a similar estimator object philosophy adapted for hybrid DAEs through the `HybridDAE` and `ProblemDefinition` entry points, reducing the learning curve for practitioners already familiar with similarly designed frameworks. In the same spirit of usability, unlike other Python packages making use of state of the art linear/nonlinear solvers (e.g. MA27 or IPOPT), SiNDAE features an entirely licensed-binary-free default installation by using the FERAL [@kitchin2026feral] and POUNCE [@kitchin2026pounce] solvers.


![Dependency graph generated with [pydeps](https://github.com/thebjorn/pydeps/) showing the packages used in ``SiNDAE``.\label{fig:fig1}](./images/sindae_pydeps.png)



# Vignette

Point to examples gallery directory in the docs.

Describe the example.

Code snippets of solve.

# Availability

Point to PyPI deployment, Github, and docs deployment.

# Acknowledgements

# References