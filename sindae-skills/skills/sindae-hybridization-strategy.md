---
name: hybridization-strategy
description: Decide how much of a mechanistic model to replace with a neural network. Use when the user is choosing what to learn, weighing degree of hybridization against data cost, extrapolation, and interpretability.
---

# Hybridization Strategy

## Degree of Hybridization

The degree of hybridization is a qualitative scale defined in Narayanan et al's paper "Hybrid Models Based on Machine Learning and an Increasing Degree of Process Knowledge: Application to Cell Culture Processes" to describe the extent to which mechanistic knowledge versus learned data-driven knowledge is incorporated into a model. The scale is defined as:

```
0% (fully data driven) <------------> 100% (purely mechanistic)
```

- A 0\% hybridized system would constitute the entire RHS of a differential / algebraic equation approximated by a neural network.
    - $\frac{dX_v}{dt} = f_{NN}(\theta)$
- A (relatively) intermediate amount of hybridization would incorporate some fundamental mechanistic knowledge and some learned components. e.g. a mechanistic mass balance and a learned reaction term / a learned reaction rate constant.
    - $\frac{dX_v}{dt} = IN - OUT + f_{NN}(\theta)$ (less hybridized)
    - $\frac{dX_v}{dt} = IN - OUT + f_{NN}(\theta)\cdot X_V$ (more hybridized)
- A 100\% hybridized model is a purely mechanistic model. e.g.
    - $\frac{dX_v}{dt} = IN - OUT + r \cdot X_V$

Moving from 0 to 100% degree of hybridization, more phenomena are explicitly described in
the hybrid model increases with the most general assumptions and phenomena are described first.
Always incorporate fundamental first principles information explicitly, learning more complex information, rather than vice versa. In a process model, this means conservation laws before kinetics, and kinetics before specific rate laws. 

When suggesting the hybridization of components of a model, the inputs to the neural network should always be measurable quantities, otherwise the model will not be practical.

## Trade-offs

When suggesting a hybridization strategy for a user, consider the following trade-offs associated with different degrees of hybridization.

## 1. Model Accuracy & Hybridization vs Cost
*Cost* is defined as the minimum number of experiments (trajectories / batches / runs) required to build an optimal version of the model
- As the degree of hybridization increases, the model cost decreases to a point before which the bias imposed by modeling assumptions becomes significant. 
- A 0% hybridized (fully data driven model) may require 50 data points to be trained appropriately, while a 70% hybridized model may only require 10 data points and achieve better predictive performance. 
- A fully mechanistic model, where all kinetic parameters are known, needs 0 data to produce predictions, though they may be very inaccurate due to the modelling bias. 
- Empirically, it is advised to include as much hybridization as possible before training / prediction error increases due to bias. 
- Suggest training and comparing multiple models accuracy with varying degrees of hybridization based on the amount of data on hand before selecting one. 
- Highly data driven models should be avoided when little data is available or when collecting new data is costly.
- Increasing the degree of hybridization is generally advised to reduce the cost of the model, as long as the introduced process knowledge is not such that a strong bias reduces accuracy.
- Increasing network width / depth / number of output nodes increases the number of parameters and therefore comes at an increased model cost due to the risk of overfitting.


## 2. Low Data Regime 
- Purely data driven (0% hybridized) models perform poorly when little data is available.
- Models with increasing amounts of process knowledge in the form of hybridization perform better to a point where incorrect modeling bias becomes significant.
- Mechanistic model performance is invariant to the number of data. This does not guarantee accurate predictive performance for complex or poorly understood processes.

## 3. Extrapolation Performance
*Extrapolation* can be defined as prediction using unseen initial conditions or predictions beyond the training data range.
- Data driven models perform moderately well on interpolation prediction problems if trained on sufficient data and with regularization. However, extrapolation performance is usually poor. Avoid suggesting data driven or lightly hybridized models when out extrapolation or prediction on
new operating conditions is the goal.
- Extrapolation prediction typically reduces with increasing process knowledge in the mechanistic model to a point.
- Increased network size / prediction outputs also requires additional data for accurate extrapolation predictions due to the risk of overfitting.
- With very large training data, highly hybridized models typically show strong performance.

## Interpretability
- The higher the degree of hybridization, the more interpretable the underlying model since terms or parameter's with a physically grounded influence on the model dynamics can be replaced with a neural network. 
- Distinguishing the mechanistic influence of the entire RHS approximated by a NN is substantially harder than a kinetic rate constant approximated by a NN. 
- If the model exists to test process hypotheses, discover the influence of measured variables on parameters, or support Quality by Design, interpretability can outrank raw accuracy.
