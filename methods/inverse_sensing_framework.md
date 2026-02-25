# Neural Operator-Based Inverse Sensing in Chaotic Multi-Object Systems

## A Mathematical Framework for State Inference from Density Observations

---

**Abstract**

We present a mathematical framework for inferring the states (number, positions, velocities, masses, and sizes) of multiple interacting objects from observations of their spatial density evolution. The key insight is that chaotic collision dynamics, rather than hindering inference, actually enable it by creating unique density evolution signatures for different initial conditions. We develop two problem formulations: (1) the *fully-observed* inverse problem where complete density fields are available at discrete times, and (2) the *partially-observed* inverse problem where density must be sampled through constrained sensor trajectories. For the fully-observed case, we employ a Fourier Neural Operator (FNO) as a differentiable forward model within gradient-free optimization, initialized via deconvolution-based state estimation. For the partially-observed case, we extend this to an active sensing framework that jointly optimizes sensor trajectories and state estimates using information-theoretic objectives. We discuss applications to defense (tracking from degraded ISR) and microscopy (super-resolution imaging of diffusing particles).

---

## Table of Contents

1. [Introduction and Motivation](#1-introduction-and-motivation)
2. [Problem Formulation](#2-problem-formulation)
3. [Forward Model: Density Evolution via FNO](#3-forward-model-density-evolution-via-fno)
4. [The Fully-Observed Inverse Problem](#4-the-fully-observed-inverse-problem)
5. [Initialization via Deconvolution](#5-initialization-via-deconvolution)
6. [Alternative Initialization Methods](#6-alternative-initialization-methods)
7. [The Partially-Observed Inverse Problem](#7-the-partially-observed-inverse-problem)
8. [Information-Theoretic Sensor Planning](#8-information-theoretic-sensor-planning)
9. [Computational Algorithms](#9-computational-algorithms)
10. [Applications](#10-applications)
11. [Conclusion](#11-conclusion)
12. [References](#12-references)

---

## 1. Introduction and Motivation

### 1.1 The Fundamental Challenge

Consider the problem of tracking multiple objects that interact through collisions or other contact dynamics. Traditional multi-object tracking assumes that individual objects can be detected and associated across time frames. However, in many practical scenarios, sensor limitations prevent individual object detection:

- **Low spatial resolution**: Objects appear as blurred density distributions rather than discrete points
- **Low temporal resolution**: Significant object motion occurs between observation frames
- **High object density**: Objects overlap in sensor observations
- **Noisy measurements**: Signal-to-noise ratio is insufficient for individual detection

In such cases, we observe an aggregate *density field* rather than individual object states. The inverse problem is: given observations of density evolution, can we infer the underlying object states?

### 1.2 The Role of Chaos

Counterintuitively, *chaotic dynamics help rather than hinder this inference*. In a chaotic system, nearby initial conditions diverge exponentially:

$$\|\delta \mathbf{x}(t)\| \sim \|\delta \mathbf{x}(0)\| e^{\lambda t}$$

where $\lambda > 0$ is the Lyapunov exponent. This sensitivity means that:

1. Different initial conditions produce *distinguishably different* density evolutions
2. Over sufficient observation time $T \gg 1/\lambda$, the density trajectory becomes a unique "fingerprint" of the initial state
3. The inverse problem, while nonlinear, is *well-posed* in the sense of having a unique solution

### 1.3 Scope and Contributions

This document presents:

1. A mathematical formulation of the inverse sensing problem for chaotic multi-object systems
2. A solution approach using Fourier Neural Operators as learned forward models
3. Initialization strategies based on deconvolution and learned mappings
4. Extension to partially-observed densities with active sensor planning
5. Connections to defense sensing and fluorescence microscopy applications

---

## 2. Problem Formulation

### 2.1 State Space

We consider $N$ interacting objects (particles) in a bounded 2D domain $\Omega \subset \mathbb{R}^2$. The complete state is:

$$\boldsymbol{\theta} = \{N, \{\mathbf{x}_i, \mathbf{v}_i, m_i, d_i\}_{i=1}^N\}$$

where for each particle $i$:
- $\mathbf{x}_i \in \Omega$ is the position
- $\mathbf{v}_i \in \mathbb{R}^2$ is the velocity
- $m_i \in \mathbb{R}_+$ is the mass
- $d_i \in \mathbb{R}_+$ is the diameter

The state space is the disjoint union over possible particle counts:

$$\Theta = \bigcup_{N=1}^{N_{\max}} \Theta_N, \quad \Theta_N = \Omega^N \times \mathbb{R}^{2N} \times \mathbb{R}_+^N \times \mathbb{R}_+^N$$

### 2.2 Density Field

The spatial density field induced by state $\boldsymbol{\theta}$ is:

$$\rho_{\boldsymbol{\theta}}(\mathbf{r}, t) = \sum_{i=1}^N K_{d_i}(\mathbf{r} - \mathbf{x}_i(t))$$

where $K_d(\cdot)$ is a kernel function representing the spatial extent of each particle. Common choices include:

**Gaussian kernel:**
$$K_d(\mathbf{r}) = \frac{1}{2\pi\sigma_d^2} \exp\left(-\frac{\|\mathbf{r}\|^2}{2\sigma_d^2}\right), \quad \sigma_d = d/2$$

**Top-hat kernel:**
$$K_d(\mathbf{r}) = \frac{1}{\pi(d/2)^2} \mathbf{1}_{\|\mathbf{r}\| \leq d/2}$$

**Normalized kernel (integrates to 1):**
$$\int_{\Omega} K_d(\mathbf{r}) d\mathbf{r} = 1$$

### 2.3 Dynamics

The particles evolve according to Hamiltonian dynamics with elastic collisions. Between collisions:

$$\dot{\mathbf{x}}_i = \mathbf{v}_i, \quad \dot{\mathbf{v}}_i = \mathbf{f}_i / m_i$$

where $\mathbf{f}_i$ represents external forces (e.g., gravity, confinement). At collision between particles $i$ and $j$:

$$\mathbf{v}_i' = \mathbf{v}_i - \frac{2m_j}{m_i + m_j}(\mathbf{v}_i - \mathbf{v}_j) \cdot \hat{\mathbf{n}} \, \hat{\mathbf{n}}$$

$$\mathbf{v}_j' = \mathbf{v}_j - \frac{2m_i}{m_i + m_j}(\mathbf{v}_j - \mathbf{v}_i) \cdot \hat{\mathbf{n}} \, \hat{\mathbf{n}}$$

where $\hat{\mathbf{n}}$ is the collision normal.

### 2.4 The Forward Problem

The forward problem maps initial state to density evolution:

$$\mathcal{F}: \boldsymbol{\theta} \mapsto \{\rho_{\boldsymbol{\theta}}(\cdot, t)\}_{t \in [0,T]}$$

This map is:
- **Deterministic**: Given $\boldsymbol{\theta}$, the density evolution is uniquely determined
- **Nonlinear**: Due to collision dynamics
- **Chaotic**: Small changes in $\boldsymbol{\theta}$ lead to large changes in $\rho$ at large $t$

### 2.5 The Inverse Problem

The inverse problem seeks to recover state from density observations:

$$\mathcal{F}^{-1}: \{\hat{\rho}(\cdot, t_k)\}_{k=0}^{K-1} \mapsto \boldsymbol{\theta}$$

where $\hat{\rho}(\cdot, t_k)$ is the observed density at time $t_k = k \cdot \Delta t$.

---

## 3. Forward Model: Density Evolution via FNO

### 3.1 The Perron-Frobenius Operator

The evolution of probability density is governed by the Perron-Frobenius (PF) operator $\mathcal{P}_t$:

$$\rho(\mathbf{r}, t) = \mathcal{P}_t[\rho(\cdot, 0)](\mathbf{r})$$

A key property: *the PF operator is linear*, even when the underlying particle dynamics are nonlinear. This linearity enables learning-based approximation.

### 3.2 Fourier Neural Operator Architecture

We approximate the PF operator using a Fourier Neural Operator (FNO). The FNO learns the mapping:

$$\rho(\cdot, t + \Delta t) = \text{FNO}_{\phi}[\rho(\cdot, t)]$$

where $\phi$ represents learned parameters.

**Architecture:**

The 2D FNO consists of:

1. **Lifting layer**: Project input density to higher-dimensional feature space
   $$\mathbf{h}_0(\mathbf{r}) = W_0 \rho(\mathbf{r}) + \mathbf{b}_0$$

2. **Fourier layers** ($L$ layers): Each layer combines spectral and local operations
   $$\mathbf{h}_{l+1} = \sigma\left(\mathcal{K}_l[\mathbf{h}_l] + W_l \mathbf{h}_l\right)$$
   
   where the spectral convolution is:
   $$\mathcal{K}_l[\mathbf{h}](\mathbf{r}) = \mathcal{F}^{-1}\left[R_l \cdot \mathcal{F}[\mathbf{h}]\right](\mathbf{r})$$
   
   Here $\mathcal{F}$ is the 2D Fourier transform and $R_l$ are learnable weights in Fourier space.

3. **Projection layer**: Map back to density space
   $$\rho_{out}(\mathbf{r}) = W_L \mathbf{h}_L(\mathbf{r}) + b_L$$

**Fourier Space Operations:**

In discrete form with grid resolution $n_x \times n_y$:

$$\mathcal{F}[\mathbf{h}]_{k_x, k_y} = \sum_{x=0}^{n_x-1} \sum_{y=0}^{n_y-1} \mathbf{h}_{x,y} \exp\left(-2\pi i \left(\frac{k_x x}{n_x} + \frac{k_y y}{n_y}\right)\right)$$

Only the first $m_x \times m_y$ modes are retained (low-pass filtering), with learnable complex weights $R_l \in \mathbb{C}^{c_{in} \times c_{out} \times m_x \times m_y}$.

### 3.3 Training the FNO

**Training Data Generation:**

Generate diverse trajectories by:
1. Sampling initial states $\boldsymbol{\theta}^{(j)} \sim p(\boldsymbol{\theta})$
2. Simulating dynamics to obtain $\rho^{(j)}(\cdot, t_k)$
3. Creating input-output pairs $(\rho^{(j)}_k, \rho^{(j)}_{k+1})$

**Loss Function:**

$$\mathcal{L}(\phi) = \mathbb{E}_{j,k}\left[\|\text{FNO}_\phi[\rho^{(j)}_k] - \rho^{(j)}_{k+1}\|^2_{L^2}\right]$$

In discrete form:
$$\mathcal{L}(\phi) = \frac{1}{|\mathcal{D}|} \sum_{(j,k) \in \mathcal{D}} \sum_{x,y} \left(\text{FNO}_\phi[\rho^{(j)}_k]_{x,y} - \rho^{(j)}_{k+1,x,y}\right)^2$$

**Multi-step Training:**

For improved long-horizon accuracy, train on multi-step predictions:

$$\mathcal{L}_{multi}(\phi) = \sum_{s=1}^{S} \gamma^{s-1} \|\text{FNO}_\phi^{(s)}[\rho_0] - \rho_s\|^2$$

where $\text{FNO}^{(s)}$ denotes $s$ sequential applications and $\gamma < 1$ is a discount factor.

### 3.4 Density Field Discretization

For computation, the density is represented on a uniform grid:

$$\rho_{i,j}(t) \approx \rho(x_i, y_j, t), \quad x_i = x_{\min} + i \cdot \Delta x, \quad y_j = y_{\min} + j \cdot \Delta y$$

The grid resolution must satisfy:
- $\Delta x, \Delta y < d_{\min}$ (resolve smallest particle)
- $n_x \cdot n_y$ compatible with FFT efficiency (powers of 2)

---

## 4. The Fully-Observed Inverse Problem

### 4.1 Problem Statement

**Given:** Sequence of fully-observed density fields $\{\hat{\rho}_k\}_{k=0}^{K-1}$ at times $t_k = k \cdot \Delta t$

**Find:** State $\boldsymbol{\theta}^* = \{N^*, \{\mathbf{x}_i^*, \mathbf{v}_i^*, m_i^*, d_i^*\}\}$ that best explains the observations

### 4.2 Objective Function

We formulate this as a nonlinear least squares problem:

$$\boldsymbol{\theta}^* = \arg\min_{\boldsymbol{\theta}} \mathcal{J}(\boldsymbol{\theta})$$

where the objective function is:

$$\mathcal{J}(\boldsymbol{\theta}) = \sum_{k=0}^{K-1} \|\text{FNO}^{(k)}[\rho_{\boldsymbol{\theta}}(\cdot, 0)] - \hat{\rho}_k\|^2_{L^2} + \mathcal{R}(\boldsymbol{\theta})$$

Here:
- $\text{FNO}^{(k)}$ denotes $k$ sequential applications of the FNO
- $\rho_{\boldsymbol{\theta}}(\cdot, 0)$ is the initial density implied by state $\boldsymbol{\theta}$
- $\mathcal{R}(\boldsymbol{\theta})$ is a regularization term

### 4.3 Regularization

The regularizer encodes prior knowledge and prevents overfitting:

$$\mathcal{R}(\boldsymbol{\theta}) = \lambda_N \cdot N + \lambda_m \sum_{i=1}^N (m_i - \bar{m})^2 + \lambda_d \sum_{i=1}^N (d_i - \bar{d})^2 + \lambda_v \sum_{i=1}^N \|\mathbf{v}_i\|^2$$

where:
- $\lambda_N \cdot N$ is an Occam's razor penalty preferring fewer particles
- $\lambda_m, \lambda_d$ penalize deviation from expected mass/diameter
- $\lambda_v$ penalizes large velocities (prior toward slow motion)

### 4.4 Optimization Strategy

The optimization is challenging due to:
1. **Discrete parameter**: $N$ is an integer
2. **Non-convexity**: Multiple local minima exist
3. **High dimensionality**: $\dim(\boldsymbol{\theta}) = 1 + 6N$ for 2D

**Two-Level Optimization:**

*Outer loop*: Search over discrete $N$ values
```
For N = N_min, ..., N_max:
    θ_N* = OptimizeContinuous(N, ρ_observed)
    J_N = Objective(θ_N*)
    
Return argmin_N J_N
```

*Inner loop*: Optimize continuous parameters for fixed $N$

### 4.5 Gradient-Free Optimization

Due to the discrete collisions in the underlying dynamics, the objective landscape may have discontinuities. We employ gradient-free methods:

**CMA-ES (Covariance Matrix Adaptation Evolution Strategy):**

CMA-ES maintains a multivariate Gaussian search distribution:
$$\boldsymbol{\theta} \sim \mathcal{N}(\mathbf{m}, \sigma^2 \mathbf{C})$$

At each generation:
1. Sample $\lambda$ candidate solutions: $\boldsymbol{\theta}^{(i)} \sim \mathcal{N}(\mathbf{m}, \sigma^2 \mathbf{C})$
2. Evaluate objective: $f^{(i)} = \mathcal{J}(\boldsymbol{\theta}^{(i)})$
3. Update mean toward best solutions: $\mathbf{m} \leftarrow \sum_{i=1}^\mu w_i \boldsymbol{\theta}^{(i:\lambda)}$
4. Adapt covariance matrix and step size

**Differential Evolution:**

Maintains a population of candidate solutions and evolves them through mutation, crossover, and selection:

$$\mathbf{u} = \boldsymbol{\theta}^{(r_1)} + F \cdot (\boldsymbol{\theta}^{(r_2)} - \boldsymbol{\theta}^{(r_3)})$$

where $F \in [0.5, 1.0]$ is the mutation factor.

### 4.6 Identifiability Analysis

The inverse problem is identifiable if different states produce distinguishable density evolutions:

$$\boldsymbol{\theta}_1 \neq \boldsymbol{\theta}_2 \implies \exists k: \rho_{\boldsymbol{\theta}_1}(\cdot, t_k) \neq \rho_{\boldsymbol{\theta}_2}(\cdot, t_k)$$

**Sufficient conditions for identifiability:**

1. **Observation time**: $T = K \cdot \Delta t \gg 1/\lambda$ (Lyapunov time)
2. **Temporal resolution**: $\Delta t$ small enough to capture collision events
3. **Spatial resolution**: Grid spacing $\Delta x, \Delta y < d_{\min}$
4. **Chaotic dynamics**: Lyapunov exponent $\lambda > 0$ (requires unequal masses)

**Quantifying identifiability:**

Define the distinguishability metric:
$$D(\boldsymbol{\theta}_1, \boldsymbol{\theta}_2) = \frac{1}{K} \sum_{k=0}^{K-1} \|\rho_{\boldsymbol{\theta}_1}(\cdot, t_k) - \rho_{\boldsymbol{\theta}_2}(\cdot, t_k)\|_{L^2}$$

The inverse problem is well-conditioned if:
$$D(\boldsymbol{\theta}_1, \boldsymbol{\theta}_2) \geq \kappa \cdot \|\boldsymbol{\theta}_1 - \boldsymbol{\theta}_2\|$$

for some $\kappa > 0$ (amplification factor).

---

## 5. Initialization via Deconvolution

Good initialization dramatically improves optimization convergence. We exploit the structure of the density field to obtain initial estimates.

### 5.1 The Deconvolution Problem

The observed density at $t=0$ is a superposition of kernels:
$$\rho(\mathbf{r}, 0) = \sum_{i=1}^N K_{d_i}(\mathbf{r} - \mathbf{x}_i) = (K * \mu)(\mathbf{r})$$

where $\mu = \sum_{i=1}^N \delta(\mathbf{r} - \mathbf{x}_i)$ is the discrete measure of particle locations.

Deconvolution seeks to recover $\mu$ from $\rho$.

### 5.2 Wiener Deconvolution

Assuming a known average kernel $\bar{K}$ and additive noise:
$$\hat{\rho} = K * \mu + \eta$$

The Wiener filter estimates $\mu$ in Fourier space:
$$\hat{\mathcal{F}}[\mu](\mathbf{k}) = \frac{\overline{\mathcal{F}[K](\mathbf{k})}}{\left|\mathcal{F}[K](\mathbf{k})\right|^2 + \text{SNR}^{-1}} \cdot \mathcal{F}[\hat{\rho}](\mathbf{k})$$

where $\text{SNR}$ is the signal-to-noise ratio.

In discrete form:
$$\hat{\mu}_{x,y} = \mathcal{F}^{-1}\left[\frac{\bar{K}^*_{k_x, k_y}}{|K_{k_x,k_y}|^2 + \epsilon} \cdot \hat{\rho}_{k_x, k_y}\right]$$

### 5.3 Peak Detection

After deconvolution, particle positions are estimated via local maximum detection:

**Algorithm: Peak Detection**
```
Input: Deconvolved image μ̂, threshold τ, minimum separation d_min

1. Apply local maximum filter: M = LocalMax(μ̂, window=d_min)
2. Find peaks: P = {(x,y) : μ̂(x,y) = M(x,y) and μ̂(x,y) > τ}
3. Non-maximum suppression: Remove peaks closer than d_min
4. Convert grid indices to physical coordinates

Output: Estimated positions {x̂_i}
```

### 5.4 Velocity Estimation via Optical Flow

Given two consecutive density frames $\rho_0, \rho_1$, velocities are estimated using optical flow.

**Lucas-Kanade Method:**

Assuming brightness constancy:
$$\rho(\mathbf{r}, t) = \rho(\mathbf{r} + \mathbf{v}\Delta t, t + \Delta t)$$

Taylor expansion yields the optical flow constraint:
$$\nabla \rho \cdot \mathbf{v} + \frac{\partial \rho}{\partial t} = 0$$

At each detected particle position $\hat{\mathbf{x}}_i$, solve for local velocity:
$$\hat{\mathbf{v}}_i = -\left(\nabla \rho \nabla \rho^T\right)^{-1} \nabla \rho \frac{\partial \rho}{\partial t} \bigg|_{\mathbf{r} = \hat{\mathbf{x}}_i}$$

**Practical computation:**

Using finite differences:
$$\frac{\partial \rho}{\partial x} \approx \frac{\rho_0(x+1, y) - \rho_0(x-1, y)}{2\Delta x}$$
$$\frac{\partial \rho}{\partial t} \approx \frac{\rho_1(x, y) - \rho_0(x, y)}{\Delta t}$$

### 5.5 Mass and Diameter Estimation

**Diameter from peak width:**

The full-width at half-maximum (FWHM) of each peak in the deconvolved image relates to particle diameter:
$$\hat{d}_i = \text{FWHM}(\hat{\mu}, \hat{\mathbf{x}}_i) / c$$

where $c$ is a calibration constant depending on the kernel shape.

**Mass from intensity:**

If density is proportional to mass:
$$\hat{m}_i \propto \int_{B_i} \hat{\rho}(\mathbf{r}) d\mathbf{r}$$

where $B_i$ is a neighborhood around particle $i$.

### 5.6 Complete Initialization Procedure

**Algorithm: Deconvolution-Based Initialization**
```
Input: Observed densities {ρ̂_k}, kernel K, grid parameters

1. Deconvolve first frame:
   μ̂ = WienerDeconvolve(ρ̂_0, K)
   
2. Detect particle positions:
   {x̂_i} = PeakDetection(μ̂)
   N̂ = |{x̂_i}|
   
3. Estimate velocities (if K ≥ 2):
   {v̂_i} = OpticalFlow(ρ̂_0, ρ̂_1, {x̂_i})
   
4. Estimate diameters:
   {d̂_i} = EstimateFWHM(μ̂, {x̂_i})
   
5. Estimate masses:
   {m̂_i} = EstimateIntensity(ρ̂_0, {x̂_i})

Output: Initial state θ̂_init = {N̂, {x̂_i, v̂_i, m̂_i, d̂_i}}
```

---

## 6. Alternative Initialization Methods

### 6.1 CNN-Based Direct Mapping

A convolutional neural network can learn to directly map density fields to particle states.

**Architecture:**

```
Input: ρ(r) ∈ ℝ^(n_x × n_y)

Encoder:
  Conv2D(1 → 32, 3×3) + ReLU + MaxPool
  Conv2D(32 → 64, 3×3) + ReLU + MaxPool
  Conv2D(64 → 128, 3×3) + ReLU + MaxPool
  Flatten → FC(128·(n/8)² → 512) + ReLU

Heads:
  Count head: FC(512 → N_max) + Softmax → P(N)
  Position head: FC(512 → 2·N_max) → {x̂_i, ŷ_i}
  Velocity head: FC(512 → 2·N_max) → {v̂_x,i, v̂_y,i}

Output: N̂, {x̂_i}, {v̂_i}
```

**Training:**

Generate paired data $({\rho^{(j)}}, \boldsymbol{\theta}^{(j)})$ and train with:
$$\mathcal{L}_{CNN} = \mathcal{L}_{count} + \mathcal{L}_{position} + \mathcal{L}_{velocity}$$

where:
- $\mathcal{L}_{count} = -\sum_N \mathbf{1}[N^{(j)} = N] \log P(N)$ (cross-entropy)
- $\mathcal{L}_{position} = \min_{\pi} \sum_i \|\hat{\mathbf{x}}_i - \mathbf{x}^{(j)}_{\pi(i)}\|^2$ (Hungarian matching)
- $\mathcal{L}_{velocity}$ similarly with matched velocities

**Advantages:**
- Very fast at inference time
- Can learn complex density-to-state mappings
- Handles variable $N$ naturally

**Disadvantages:**
- Requires large training dataset
- May not generalize to out-of-distribution densities
- Less interpretable than deconvolution

### 6.2 Gaussian Mixture Model Fitting

Model the density as a Gaussian Mixture Model (GMM):
$$\rho(\mathbf{r}) \approx \sum_{i=1}^N w_i \mathcal{N}(\mathbf{r}; \boldsymbol{\mu}_i, \Sigma_i)$$

Fit via Expectation-Maximization (EM):
- E-step: Compute responsibilities $\gamma_{ij} = p(z_j = i | \mathbf{r}_j)$
- M-step: Update $w_i, \boldsymbol{\mu}_i, \Sigma_i$

**Model selection** for $N$ via BIC:
$$\text{BIC}(N) = -2 \log L + k \log n$$

where $k = N(1 + 2 + 3) - 1 = 6N - 1$ is the number of parameters.

### 6.3 Sparse Recovery Methods

Formulate position recovery as sparse optimization:
$$\min_{\mathbf{c}} \|\mathbf{c}\|_1 \quad \text{s.t.} \quad \|A\mathbf{c} - \hat{\boldsymbol{\rho}}\|_2 \leq \epsilon$$

where:
- $\mathbf{c} \in \mathbb{R}^{n_x n_y}$ is a sparse coefficient vector
- $A$ is the dictionary matrix with kernels centered at each grid point
- Non-zero entries of $\mathbf{c}$ indicate particle locations

Solve via LASSO, Orthogonal Matching Pursuit, or ADMM.

### 6.4 Comparison of Methods

| Method | Speed | Accuracy | Robustness to Noise | Handles Variable N |
|--------|-------|----------|---------------------|-------------------|
| Wiener Deconvolution | Fast | Moderate | Low | Yes (threshold) |
| CNN Direct Mapping | Very Fast | High | High | Yes |
| GMM + EM | Moderate | High | Moderate | Via model selection |
| Sparse Recovery | Slow | High | High | Yes |

**Recommendation:** Use Wiener deconvolution + peak detection for initial estimate (fast, interpretable), then refine with optimization. Use CNN for real-time applications.

---

## 7. The Partially-Observed Inverse Problem

### 7.1 Sensor Model

In many applications, the full density field cannot be observed instantaneously. Instead, a sensor samples the density along a trajectory.

**Point sensor:**
$$y(t) = \rho(\mathbf{s}(t), t) + \eta(t)$$

where $\mathbf{s}(t) \in \Omega$ is the sensor position and $\eta(t) \sim \mathcal{N}(0, \sigma^2)$ is measurement noise.

**Line sensor:**
$$y(t) = \int_0^L \rho(\mathbf{s}(t) + \lambda \hat{\mathbf{u}}(t), t) d\lambda + \eta(t)$$

where $\hat{\mathbf{u}}(t)$ is the scan direction and $L$ is the scan length.

**General sensor with point spread function:**
$$y(t) = \int_\Omega W(\mathbf{r} - \mathbf{s}(t)) \rho(\mathbf{r}, t) d\mathbf{r} + \eta(t)$$

where $W(\cdot)$ is the sensor PSF.

### 7.2 Sensor Constraints

The sensor trajectory is subject to physical constraints:

**Velocity constraint:**
$$\|\dot{\mathbf{s}}(t)\| \leq v_{\max}$$

**Acceleration constraint:**
$$\|\ddot{\mathbf{s}}(t)\| \leq a_{\max}$$

**Domain constraint:**
$$\mathbf{s}(t) \in \Omega \quad \forall t$$

### 7.3 Partial Observation Model

Over time interval $[0, T]$, the sensor collects measurements:
$$\mathcal{Y}_T = \{(y_k, \mathbf{s}_k, t_k)\}_{k=1}^K$$

This provides *partial* information about the density evolution:
- Only sampled at locations $\{\mathbf{s}_k\}$
- The density field is never fully observed at any instant

### 7.4 Modified Inverse Problem

**Given:** Partial observations $\mathcal{Y}_T$ and sensor trajectory $\{\mathbf{s}(t)\}_{t \in [0,T]}$

**Find:** State $\boldsymbol{\theta}^*$ that best explains the observations

**Modified objective:**
$$\mathcal{J}_{partial}(\boldsymbol{\theta}) = \sum_{k=1}^K \left(y_k - h(\mathbf{s}_k, \rho_{\boldsymbol{\theta}}(\cdot, t_k))\right)^2 + \mathcal{R}(\boldsymbol{\theta})$$

where $h(\mathbf{s}, \rho)$ is the sensor observation model.

### 7.5 Reconstructing Density from Partial Observations

Before solving the inverse problem, we may want to reconstruct the full density field from partial observations.

**Spatial interpolation (single time):**

Given measurements $\{(y_k, \mathbf{s}_k)\}$ at time $t$, estimate $\hat{\rho}(\mathbf{r}, t)$ via:
- Gaussian Process regression
- Radial basis function interpolation
- Sparse reconstruction with physics prior

**Spatio-temporal reconstruction:**

Using the FNO as a prior, reconstruct density by solving:
$$\min_{\{\rho_k\}} \sum_{k=1}^{K-1} \|\rho_{k+1} - \text{FNO}[\rho_k]\|^2 + \lambda \sum_k \sum_{j: t_j = t_k} (y_j - \rho_k(\mathbf{s}_j))^2$$

This enforces temporal consistency via the learned dynamics.

### 7.6 Coupling Between Sensing and Inference

A key insight: **the sensor trajectory affects inference quality**. Some trajectories provide more information about $\boldsymbol{\theta}$ than others.

This leads to the *active sensing* problem: jointly optimize the sensor trajectory and state estimate.

---

## 8. Information-Theoretic Sensor Planning

### 8.1 The Active Sensing Problem

**Objective:** Choose sensor trajectory $\mathbf{s}(\cdot)$ to maximize information about state $\boldsymbol{\theta}$

**Formulation:**
$$\mathbf{s}^*(\cdot) = \arg\max_{\mathbf{s}(\cdot)} I(\mathcal{Y}_T; \boldsymbol{\theta} | \mathbf{s}(\cdot))$$

subject to sensor dynamics constraints.

Here $I(\cdot; \cdot)$ denotes mutual information.

### 8.2 Belief State Representation

At time $t$, the *belief state* is the posterior distribution over $\boldsymbol{\theta}$:
$$b_t(\boldsymbol{\theta}) = p(\boldsymbol{\theta} | \mathcal{Y}_t)$$

We represent this using a particle filter:
$$b_t \approx \sum_{j=1}^M w_t^{(j)} \delta(\boldsymbol{\theta} - \boldsymbol{\theta}^{(j)})$$

where $\{\boldsymbol{\theta}^{(j)}, w_t^{(j)}\}_{j=1}^M$ are weighted particles.

### 8.3 Belief Update (Bayesian Filtering)

When a new observation $(y, \mathbf{s})$ arrives:

**Likelihood:**
$$p(y | \boldsymbol{\theta}, \mathbf{s}) = \mathcal{N}(y; h(\mathbf{s}, \rho_{\boldsymbol{\theta}}), \sigma^2)$$

**Weight update:**
$$w^{(j)}_{new} \propto w^{(j)}_{old} \cdot p(y | \boldsymbol{\theta}^{(j)}, \mathbf{s})$$

**Prediction step** (if particles have dynamics):
$$\boldsymbol{\theta}^{(j)}_{t+1} \sim p(\boldsymbol{\theta}_{t+1} | \boldsymbol{\theta}^{(j)}_t)$$

For our problem, the particle state $\boldsymbol{\theta}$ is constant (initial conditions), but the *predicted density* evolves:
$$\rho^{(j)}_{t+1} = \text{FNO}[\rho^{(j)}_t]$$

### 8.4 Information Gain

The information gain from a measurement at location $\mathbf{s}$ is:

$$I(\mathbf{s}) = H[p(\boldsymbol{\theta} | \mathcal{Y}_t)] - \mathbb{E}_{y|\mathbf{s}}[H[p(\boldsymbol{\theta} | \mathcal{Y}_t, y)]]$$

**Particle filter approximation:**

Current entropy:
$$H_{prior} = -\sum_j w^{(j)} \log w^{(j)}$$

Expected posterior entropy (Monte Carlo):
$$H_{posterior} \approx \frac{1}{S} \sum_{s=1}^S H[\{w^{(j)}_{new}(y_s)\}]$$

where $y_s$ are sampled from the predictive distribution.

Information gain:
$$I(\mathbf{s}) \approx H_{prior} - H_{posterior}$$

### 8.5 Information Density

Define the *information density* $\Phi(\mathbf{r}, t)$ as the expected information gain from measuring at location $\mathbf{r}$ at time $t$:

$$\Phi(\mathbf{r}, t) = I(\mathbf{s} = \mathbf{r}, t)$$

Alternatively, use the *variance-based proxy*:
$$\Phi(\mathbf{r}, t) = \text{Var}_{b_t}[\rho(\mathbf{r}, t)] = \sum_j w^{(j)} \left(\rho^{(j)}(\mathbf{r}, t) - \bar{\rho}(\mathbf{r}, t)\right)^2$$

where $\bar{\rho}(\mathbf{r}, t) = \sum_j w^{(j)} \rho^{(j)}(\mathbf{r}, t)$ is the expected density.

**Interpretation:** High $\Phi(\mathbf{r}, t)$ indicates high uncertainty about the density at $(\mathbf{r}, t)$ — measuring there provides more information.

### 8.6 Sensor Planning Strategies

**Greedy (Myopic) Planning:**

At each decision point, move toward the location of maximum information gain:
$$\mathbf{s}_{next} = \arg\max_{\mathbf{s}: \|\mathbf{s} - \mathbf{s}_{current}\| \leq v_{max} \Delta t} I(\mathbf{s})$$

**Receding Horizon Planning:**

Optimize a trajectory over horizon $\tau$:
$$\mathbf{s}^*_{t:t+\tau} = \arg\max_{\mathbf{s}(\cdot)} \int_t^{t+\tau} I(\mathbf{s}(t'), t') dt'$$

subject to dynamics constraints. Execute first segment, then replan.

**Ergodic Planning:**

Match the time-averaged sensor coverage to the information density:
$$\min_{\mathbf{s}(\cdot)} \mathcal{E}[\mathbf{s}] = \sum_{\mathbf{k}} \Lambda_{\mathbf{k}} |c_{\mathbf{k}}[\mathbf{s}] - \phi_{\mathbf{k}}|^2$$

where:
- $c_{\mathbf{k}}[\mathbf{s}]$ = Fourier coefficients of time-averaged sensor trajectory
- $\phi_{\mathbf{k}}$ = Fourier coefficients of information density $\Phi$
- $\Lambda_{\mathbf{k}} = (1 + \|\mathbf{k}\|^2)^{-s}$ weights lower frequencies more

### 8.7 Hypothesis Discrimination

When the belief has multiple distinct modes (e.g., $N=5$ vs $N=6$), prioritize measurements that *discriminate* between hypotheses.

**Discrimination score:**
$$D(\mathbf{s}) = \sum_{j_1, j_2: j_1 \neq j_2} w^{(j_1)} w^{(j_2)} |h(\mathbf{s}, \rho^{(j_1)}) - h(\mathbf{s}, \rho^{(j_2)})|^2$$

This rewards measurements where competing hypotheses predict different observations.

**Unified objective:**
$$U(\mathbf{s}) = \alpha \cdot I(\mathbf{s}) + (1-\alpha) \cdot D(\mathbf{s})$$

with $\alpha$ adaptive based on belief entropy (explore when uncertain, discriminate when multimodal).

---

## 9. Computational Algorithms

### 9.1 Algorithm: Fully-Observed Inverse Sensing

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Algorithm 1: FNO-in-Loop Inverse Sensing (Fully Observed)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Input: Observed densities {ρ̂_k}_{k=0}^{K-1}, trained FNO, config
Output: Estimated state θ* = {N*, {x_i*, v_i*, m_i*, d_i*}}

1. INITIALIZATION
   ├─ μ̂ ← WienerDeconvolve(ρ̂_0, K_avg)
   ├─ {x̂_i} ← PeakDetection(μ̂, threshold=τ)
   ├─ N_init ← |{x̂_i}|
   ├─ {v̂_i} ← OpticalFlow(ρ̂_0, ρ̂_1, {x̂_i})
   ├─ {m̂_i} ← m_default · ones(N_init)
   └─ {d̂_i} ← d_default · ones(N_init)

2. OUTER LOOP: Search over N
   ├─ For N = max(1, N_init - 2) to N_init + 2:
   │   │
   │   ├─ 2a. Adjust initial guess for N particles
   │   │      θ_init(N) ← AdjustForN({x̂_i, v̂_i, m̂_i, d̂_i}, N)
   │   │
   │   ├─ 2b. Define objective function
   │   │      J(θ) = Σ_k ||FNO^(k)[ρ_θ(0)] - ρ̂_k||² + R(θ)
   │   │
   │   ├─ 2c. INNER LOOP: Gradient-free optimization
   │   │      θ_N* ← CMA-ES(J, θ_init(N), bounds, max_iter)
   │   │
   │   └─ 2d. Record result
   │          J_N ← J(θ_N*)

3. MODEL SELECTION
   ├─ N* ← argmin_N (J_N + λ_N · N)
   └─ θ* ← θ_{N*}*

4. UNCERTAINTY QUANTIFICATION (Optional)
   └─ Σ ← HessianApprox(J, θ*) or BootstrapCovariance

Return θ*, Σ
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 9.2 Algorithm: Partially-Observed Active Inverse Sensing

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Algorithm 2: Active Inverse Sensing (Partially Observed)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Input: Prior p(θ), trained FNO, sensor constraints, time budget T
Output: Estimated state θ*, observations Y

1. INITIALIZATION
   ├─ Sample initial particles: {θ^(j)}_{j=1}^M ~ p(θ)
   ├─ Initialize weights: w^(j) ← 1/M
   ├─ Initialize sensor: s_0 ← center of domain
   └─ Observations: Y ← ∅

2. MAIN LOOP: For t = 0, Δt, 2Δt, ..., T:
   │
   ├─ 2a. PREDICT: Evolve density predictions
   │      For j = 1, ..., M:
   │          ρ^(j)_t ← FNO^(t/Δt)[ρ_{θ^(j)}(0)]
   │
   ├─ 2b. PLAN: Compute information density
   │      Φ(r, t) ← Var_{j,w}[ρ^(j)(r, t)]
   │      
   │      Option A (Greedy):
   │          s_next ← argmax_{||s - s_t|| ≤ v_max Δt} Φ(s, t)
   │      
   │      Option B (Ergodic):
   │          s_{t:t+τ} ← ErgodicPlanner(Φ, s_t, τ)
   │          s_next ← s_{t+Δt}
   │
   ├─ 2c. OBSERVE: Execute measurement
   │      Move sensor: s_t ← s_next
   │      Measure: y_t ← h(s_t, ρ_true(t)) + η
   │      Record: Y ← Y ∪ {(y_t, s_t, t)}
   │
   ├─ 2d. UPDATE: Bayesian filtering
   │      For j = 1, ..., M:
   │          ŷ^(j) ← h(s_t, ρ^(j)_t)
   │          ℓ^(j) ← N(y_t; ŷ^(j), σ²)
   │          w^(j) ← w^(j) · ℓ^(j)
   │      Normalize: w ← w / sum(w)
   │
   └─ 2e. RESAMPLE (if needed)
          If ESS(w) < M/2:
              {θ^(j), w^(j)} ← Resample({θ^(j), w^(j)})
              Optional: MCMC rejuvenation

3. FINAL ESTIMATION
   ├─ Point estimate: θ̂ ← Σ_j w^(j) θ^(j)
   ├─ Uncertainty: Σ ← Σ_j w^(j) (θ^(j) - θ̂)(θ^(j) - θ̂)^T
   │
   └─ Optional refinement:
          θ* ← LocalOptimize(J_partial, θ̂, Y)

Return θ*, Σ, Y
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 9.3 Ergodic Trajectory Optimization

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Algorithm 3: Ergodic Sensor Planning
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Input: Information density Φ(r), current position s_0, horizon τ
Output: Optimal trajectory s(t) for t ∈ [0, τ]

1. COMPUTE TARGET FOURIER COEFFICIENTS
   φ_k = (1/|Ω|) ∫_Ω Φ(r) exp(-i k·r) dr
   for |k_x| ≤ K_max, |k_y| ≤ K_max

2. PARAMETERIZE TRAJECTORY
   s(t) = s_0 + Σ_{n=1}^{N_basis} [a_n cos(ωn t) + b_n sin(ωn t)]
   
   Decision variables: {a_n, b_n}

3. TRAJECTORY FOURIER COEFFICIENTS
   c_k[s] = (1/τ) ∫_0^τ exp(-i k·s(t)) dt

4. ERGODIC METRIC
   E[s] = Σ_k Λ_k |c_k[s] - φ_k|²
   where Λ_k = (1 + ||k||²)^(-3/2)

5. OPTIMIZE
   {a_n*, b_n*} ← minimize E[s]
   subject to:
       ||ṡ(t)|| ≤ v_max ∀t
       s(t) ∈ Ω ∀t

Return s*(t) defined by {a_n*, b_n*}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 10. Applications

### 10.1 Defense: Tracking from Degraded ISR

**Scenario:** Satellite or radar provides low-resolution observations of a region containing multiple vehicles (convoy, fleet, or force elements). Individual vehicles cannot be resolved, but aggregate density patterns are observable.

**Mapping to Framework:**

| Framework Element | Defense Application |
|-------------------|---------------------|
| Particles | Vehicles, aircraft, or personnel |
| Density field | Radar returns, satellite imagery intensity |
| Sensor trajectory | Satellite orbit, radar scan pattern |
| Sensor constraints | Orbit mechanics, beam steering limits |
| Observation noise | Atmospheric effects, clutter |
| Chaotic dynamics | Traffic interactions, collision avoidance |

**Specific Applications:**

1. **Convoy tracking**: Estimate vehicle count, speeds, and types from overhead imagery where individual vehicles blur together

2. **Airfield monitoring**: Infer aircraft count and activity state from radar that cannot resolve individual aircraft

3. **Maritime domain awareness**: Track vessel formations from over-the-horizon radar with poor angular resolution

4. **Force estimation**: Determine adversary force size and disposition from degraded ISR during contested operations

**Partial Knowledge Scenario (Wargaming):**

When friendly force positions are known via blue force tracking:
- Subtract known contributions from observed density
- Infer only unknown (adversary) states
- Account for interactions between known and unknown objects

**Value Proposition:**

- Extract information from sensors previously considered too degraded
- Enable tracking when individual detection is impossible
- Provide uncertainty quantification critical for military decision-making
- Complement rather than replace traditional tracking systems

### 10.2 Fluorescence Microscopy: Super-Resolution Particle Tracking

**Scenario:** Tracking fluorescent particles (proteins, vesicles, organelles) in live cells using confocal, light-sheet, or line-scanning microscopy. Particle density exceeds optical resolution limits, and particle motion is comparable to scan time.

**Mapping to Framework:**

| Framework Element | Microscopy Application |
|-------------------|------------------------|
| Particles | Fluorescent molecules, vesicles, organelles |
| Density field | Fluorescence intensity |
| Sensor trajectory | Laser scan pattern |
| Sensor constraints | Scan speed, photodamage limits |
| Observation noise | Shot noise, background fluorescence |
| Chaotic dynamics | Brownian motion + collisions in confined geometries |

**Microscopy Modalities:**

**Confocal microscopy:**
- Point sensor with Gaussian PSF
- Raster scan over 2D plane
- Frame time: 0.1-10 seconds
- PSF width: ~200-300 nm

**Line-scanning microscopy:**
- Line sensor (1D integral of density)
- Scan perpendicular to line
- Faster than point scanning
- Enables higher temporal resolution

**Light-sheet microscopy:**
- Illuminates a plane
- Camera captures 2D projection
- Fast but limited to planar observation

**The Resolution-Speed Tradeoff:**

Traditional microscopy faces a fundamental tradeoff:
- **High spatial resolution** requires slow scanning (Nyquist sampling)
- **High temporal resolution** requires fast scanning (undersampling)

**Our Framework Breaks This Tradeoff:**

By using learned dynamics (FNO) as a prior:
1. **Undersample** spatially (faster scanning)
2. **Reconstruct** full density using temporal consistency
3. **Infer** particle states beyond optical resolution limit

**Informed Sampling Strategy:**

Rather than uniform raster scanning:
1. Compute information density $\Phi(\mathbf{r}, t)$ from current belief
2. Direct scan to high-uncertainty regions
3. Adapt scan pattern based on observed dynamics

**Specific Advantages:**

1. **Super-resolution**: Infer particle positions below diffraction limit by exploiting temporal information

2. **Reduced photodamage**: Fewer photons needed when sampling is information-directed

3. **Faster imaging**: Skip low-information regions, focus on where particles might be

4. **Dense tracking**: Track more particles than possible with single-molecule localization

**Biological Applications:**

- Vesicle trafficking in neurons
- Protein diffusion in membranes  
- Organelle dynamics during cell division
- Drug delivery particle tracking

### 10.3 Connecting Defense and Microscopy

Despite vastly different scales, these applications share key characteristics:

| Aspect | Defense | Microscopy |
|--------|---------|------------|
| Scale | km | μm |
| Object count | 10-1000 | 10-10000 |
| Sensor type | Radar, satellite | Laser scanner |
| Resolution limit | Beam width, orbit | Diffraction |
| Dynamics | Traffic, navigation | Diffusion, active transport |
| Chaos source | Collision avoidance | Particle collisions, crowding |
| Scan time vs motion | Hours vs minutes | Seconds vs milliseconds |

**Unified Mathematical Framework:**

Both applications are instances of:
- Partially observable density inference
- With chaotic multi-object dynamics
- Where sensor constraints necessitate active planning
- And learned forward models enable super-resolution inference

The mathematical machinery developed here—FNO-based forward models, deconvolution initialization, information-theoretic sensor planning—applies equally to both domains.

---

## 11. Conclusion

### 11.1 Summary of Contributions

This document has presented a comprehensive mathematical framework for inferring multi-object states from density observations in chaotic systems. Key contributions include:

1. **Problem Formulation**: Rigorous definition of the inverse sensing problem for chaotic multi-object systems, including fully-observed and partially-observed variants

2. **Forward Model**: Use of Fourier Neural Operators to learn density evolution, providing a differentiable forward model for optimization

3. **Inverse Solution**: Gradient-free optimization approach with deconvolution-based initialization, enabling robust state inference from density observations

4. **Active Sensing**: Information-theoretic framework for sensor trajectory optimization, including ergodic search and hypothesis discrimination strategies

5. **Applications**: Connections to defense ISR and fluorescence microscopy, demonstrating the broad applicability of the framework

### 11.2 Key Insights

1. **Chaos enables inference**: Chaotic dynamics create unique density evolution fingerprints, making the inverse problem well-posed over sufficient observation time

2. **Learned dynamics as regularizer**: The FNO provides a strong prior that constrains solutions to physically plausible states

3. **Information-directed sensing**: Adaptive sensor trajectories dramatically improve inference efficiency compared to uniform sampling

4. **Physics + learning synergy**: Combining physical modeling (collision dynamics) with learning (FNO) outperforms either alone

### 11.3 Future Directions

1. **3D extension**: Extend to volumetric density inference with 3D scanning

2. **Online learning**: Adapt the FNO during sensing to improve model fidelity

3. **Multi-modal sensing**: Combine density measurements with sparse particle detections

4. **Real-time implementation**: GPU-accelerated algorithms for live applications

5. **Experimental validation**: Test framework on real microscopy and radar data

---

## 12. References

### Neural Operators
- Li, Z., et al. (2020). Fourier Neural Operator for Parametric Partial Differential Equations. NeurIPS.
- Lu, L., et al. (2021). Learning nonlinear operators via DeepONet. Nature Machine Intelligence.

### Inverse Problems
- Kaipio, J. & Somersalo, E. (2005). Statistical and Computational Inverse Problems. Springer.
- Tarantola, A. (2005). Inverse Problem Theory and Methods for Model Parameter Estimation. SIAM.

### Active Sensing and Ergodic Search
- Mathew, G. & Mezić, I. (2011). Metrics for ergodicity and design of ergodic dynamics for multi-agent systems. Physica D.
- Miller, L. & Murphey, T. (2016). Trajectory optimization for continuous ergodic exploration. ACC.

### Particle Tracking
- Jaqaman, K., et al. (2008). Robust single-particle tracking in live-cell time-lapse sequences. Nature Methods.
- Chenouard, N., et al. (2014). Objective comparison of particle tracking methods. Nature Methods.

### Deconvolution
- Richardson, W.H. (1972). Bayesian-based iterative method of image restoration. JOSA.
- Lucy, L.B. (1974). An iterative technique for the rectification of observed distributions. AJ.

### Chaotic Dynamics
- Ott, E. (2002). Chaos in Dynamical Systems. Cambridge University Press.
- Sinai, Y.G. (1970). Dynamical systems with elastic reflections. Russian Mathematical Surveys.

---

## Appendix A: Notation Summary

| Symbol | Meaning |
|--------|---------|
| $\Omega$ | Spatial domain |
| $\boldsymbol{\theta}$ | Complete state (N, positions, velocities, masses, diameters) |
| $N$ | Number of particles |
| $\mathbf{x}_i$ | Position of particle $i$ |
| $\mathbf{v}_i$ | Velocity of particle $i$ |
| $m_i$ | Mass of particle $i$ |
| $d_i$ | Diameter of particle $i$ |
| $\rho(\mathbf{r}, t)$ | Density field |
| $\hat{\rho}$ | Observed density |
| $K_d(\cdot)$ | Particle kernel |
| $\mathbf{s}(t)$ | Sensor position |
| $y(t)$ | Sensor measurement |
| $h(\mathbf{s}, \rho)$ | Observation model |
| $\mathcal{F}$ | Forward map / Fourier transform |
| $\text{FNO}$ | Fourier Neural Operator |
| $b_t$ | Belief state |
| $\Phi(\mathbf{r}, t)$ | Information density |
| $I(\cdot)$ | Information gain / mutual information |
| $\mathcal{J}$ | Objective function |
| $\mathcal{R}$ | Regularizer |
| $\lambda$ | Lyapunov exponent |

---

## Appendix B: Computational Complexity

| Operation | Complexity | Notes |
|-----------|------------|-------|
| FNO forward pass | $O(n_x n_y \log(n_x n_y))$ | FFT-dominated |
| FNO K-step rollout | $O(K \cdot n_x n_y \log(n_x n_y))$ | Sequential |
| Deconvolution | $O(n_x n_y \log(n_x n_y))$ | Wiener filter |
| Peak detection | $O(n_x n_y)$ | Local maximum filter |
| Particle filter update | $O(M)$ | M particles |
| Information density | $O(M \cdot n_x n_y)$ | Variance computation |
| CMA-ES iteration | $O(\lambda \cdot K \cdot \text{FNO})$ | λ samples evaluated |
| Ergodic optimization | $O(K_{max}^2 \cdot N_{basis})$ | Fourier mode matching |

Typical values: $n_x, n_y \sim 64-256$, $M \sim 100-1000$, $K \sim 100-1000$.

---

*Document version: 1.0*  
*Last updated: 2024*