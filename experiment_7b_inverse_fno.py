"""
Experiment 7b: FNO-In-The-Loop Inverse Problem (Framing A)
===========================================================

Foil to experiment_7_inverse_problem (Framing B: collision events → masses/diameters).

**This experiment (Framing A):**
  Given an observed density-field time series ρ(0), ρ(dt), …, ρ(T), infer the
  initial particle positions and velocities using the trained FourierNeuralOperator
  from Experiment 3 as a *fast surrogate forward model* inside a gradient-free
  CMA-ES optimisation loop.

**Key insight (observability argument):**
  The map  θ = {xᵢ, vᵢ} → {ρ(0), ρ(dt), …, ρ(T)}  becomes injective as T grows.
  For a chaotic system (Lyapunov exponent λ > 0), two nearby initial conditions
  diverge as exp(λt), so their density trajectories diverge rapidly and the
  inverse problem becomes well-conditioned after  T ≈ T_Lyap = 1/λ.

**Cramér-Rao Bound (CRB) analysis:**
  We compute the Fisher Information Matrix (FIM) as a function of the observation
  horizon T and derive the minimum achievable estimation variance. This links the
  CRB decay rate to the Lyapunov exponent, showing T* ~ 1–3 × T_Lyap.

Sections
--------
1.  Setup & FNO loading
2.  Build observed density sequences from test trajectories
3.  CMA-ES objective (FNO surrogate and physics sim)
4.  Multi-trial optimisation loop
5.  Evaluation vs observation horizon T
6.  Cramér-Rao Bound analysis
7.  Results compilation and summary

Usage
-----
    python experiment_7b_inverse_fno.py                    # full run
    python experiment_7b_inverse_fno.py --smoke-test       # fast 60-second test
    python experiment_7b_inverse_fno.py --output-dir /tmp/results
"""

import argparse
import json
import os
import sys
import time
import warnings
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── torch ──────────────────────────────────────────────────────────────────────
import torch
import torch.nn.functional as F

# ── local imports ──────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fno_pytorch import (
    DensityDataset,
    DensityTrajectory,
    FourierNeuralOperator,
)
from particle_system import ParticleSystem, create_random_system

# ── optional CMA-ES ────────────────────────────────────────────────────────────
try:
    import cma  # pip install cma
    _HAS_CMA = True
except ImportError:
    _HAS_CMA = False
    warnings.warn(
        "cma package not found; falling back to scipy.optimize.differential_evolution. "
        "Install with: pip install cma",
        ImportWarning,
    )

from scipy.optimize import differential_evolution


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class InverseFNOConfig:
    """Configuration for Experiment 7b."""

    # --- Paths ---
    output_dir: str = "./results/paper1"
    checkpoint_name: str = "fno_checkpoint.pt"

    # --- System parameters (must match ExperimentConfig used in Exp 3) ---
    track_length: float = 1000.0
    n_density_bins: int = 128
    save_interval: float = 0.1   # Δt used during data generation
    # --- FNO architecture (must match ExperimentConfig used in Exp 3) ---
    fno_modes: int = 32
    fno_hidden_channels: int = 64
    fno_layers: int = 4
    k_history: int = 4     # Stacked history frames — must match trained model
    lstm_hidden: int = 64  # LSTM hidden dim
    lstm_layers: int = 1   # Number of LSTM layers

    # --- Inverse problem setup ---
    n_test_systems: int = 15       # Test trajectories to invert
    n_particles_fixed: Optional[int] = None  # None = read from traj; int = fix
    speed_range: Tuple[float, float] = (20.0, 80.0)

    # --- Observation horizon sweep ---
    # T_steps values to test; None = auto (based on trajectory length)
    horizon_steps: Optional[List[int]] = None

    # --- CMA-ES / optimiser ---
    n_trials: int = 5              # Independent restarts per system × horizon
    cmaes_maxiter: int = 500
    cmaes_sigma0: float = 0.15     # Initial std (in [0,1] normalised space)
    diff_evo_maxiter: int = 300    # Fallback if cma not installed
    diff_evo_popsize: int = 20

    # --- CRB analysis ---
    crb_n_systems: int = 5         # Systems used for FIM computation (slow)
    fim_eps: float = 1e-3          # Finite difference step for Jacobian

    # --- Noise model for FIM ---
    # If None, uses FNO val loss from checkpoint as σ²
    observation_sigma2: Optional[float] = None

    # --- Misc ---
    seed: int = 42
    smoke_test: bool = False       # If True, override to tiny config


def smoke_test_config(cfg: InverseFNOConfig) -> InverseFNOConfig:
    """Override config for a quick smoke test (< 60 s)."""
    cfg.n_test_systems = 2
    cfg.n_trials = 2
    cfg.horizon_steps = [1, 5]
    cfg.cmaes_maxiter = 50
    cfg.diff_evo_maxiter = 50
    cfg.crb_n_systems = 1
    cfg.smoke_test = True
    return cfg


# =============================================================================
# 1.  FNO Loading
# =============================================================================

def load_fno(
    cfg: InverseFNOConfig,
    device: str,
) -> Tuple[FourierNeuralOperator, float]:
    """
    Load the trained FNO checkpoint from Experiment 3.

    If the checkpoint contains a 'config' dict (saved by train_fno), the
    architecture hyper-parameters are read from it so that they always
    match the saved weights even if InverseFNOConfig was edited.

    Returns
    -------
    fno      : loaded and eval()-ed model
    val_loss : best validation loss stored in checkpoint (used as σ² floor)
    """
    checkpoint_path = os.path.join(cfg.output_dir, cfg.checkpoint_name)

    val_loss = 1e-3  # fallback

    # Architecture kwargs — prefer checkpoint config over InverseFNOConfig
    arch_kwargs = dict(
        n_bins=cfg.n_density_bins,
        n_modes=cfg.fno_modes,
        hidden_channels=cfg.fno_hidden_channels,
        n_layers=cfg.fno_layers,
        delta_t=cfg.save_interval,
        k_history=cfg.k_history,
        lstm_hidden=cfg.lstm_hidden,
        lstm_layers=cfg.lstm_layers,
    )

    if os.path.exists(checkpoint_path):
        print(f"  Loading FNO checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)

        # Override architecture from checkpoint if stored
        if "config" in ckpt:
            saved = ckpt["config"]
            for key in ("n_modes", "hidden_channels", "n_layers",
                        "k_history", "lstm_hidden", "lstm_layers"):
                if key in saved:
                    arch_kwargs[key] = saved[key]
            print(f"  Arch from checkpoint: k_history={arch_kwargs['k_history']}, "
                  f"lstm_hidden={arch_kwargs['lstm_hidden']}")

        fno = FourierNeuralOperator(**arch_kwargs)
        fno.load_state_dict(ckpt["model_state_dict"])

        if "val_loss" in ckpt:
            val_loss = float(ckpt["val_loss"])
            print(f"  Checkpoint val_loss (used as σ²): {val_loss:.6f}")
        else:
            print("  val_loss not in checkpoint; using fallback σ² = 1e-3")
    else:
        warnings.warn(
            f"FNO checkpoint not found at {checkpoint_path}. "
            "Using a randomly initialised FNO — results will be poor until "
            "experiment_3_fno_training has been run.",
            UserWarning,
        )
        fno = FourierNeuralOperator(**arch_kwargs)

    fno = fno.to(device)
    fno.eval()
    return fno, val_loss


# =============================================================================
# 2.  Dataset normalisation statistics
# =============================================================================

def build_normalisation_stats(
    dataset: Dict,
    cfg: InverseFNOConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reconstruct DensityDataset normalization mean/std from the training split.

    k_history must match what was used during Experiment 3 training so that
    the per-bin statistics are identical (means are averaged over all k frames).
    Returns (input_mean, input_std) each of shape (n_bins,).
    """
    train_trajs = []
    for traj in dataset["train"]:
        dt = DensityTrajectory(
            time_points=np.array(traj["time_points"]),
            density_fields=np.array(traj["density_fields"]),
            delta_t=cfg.save_interval,
            system_params={"n_particles": traj["n_particles"]},
        )
        train_trajs.append(dt)

    ds = DensityDataset(train_trajs, n_steps_ahead=1, normalize=True,
                        k_history=cfg.k_history)
    # input_mean has shape (1, n_bins) — squeeze to (n_bins,) for broadcasting
    mean = ds.input_mean.squeeze()   # (n_bins,)
    std  = ds.input_std.squeeze()    # (n_bins,)
    return mean, std


# =============================================================================
# 3.  Density field from particle positions
# =============================================================================

def positions_to_density(
    positions: np.ndarray,     # (n,) particle positions on [0, track_length)
    track_length: float,
    n_bins: int,
    diameters: Optional[np.ndarray] = None,   # (n,) — uses default if None
) -> np.ndarray:
    """
    Compute the coarse-grained 1-D density field matching ParticleSystem's
    compute_density_field method (Gaussian kernel, σ = diameter / 2).

    Returns
    -------
    density : (n_bins,) array
    """
    n = len(positions)
    if diameters is None:
        # Use a representative diameter (mid-range)
        diameters = np.full(n, 12.5)

    bin_centers = np.linspace(0, track_length, n_bins, endpoint=False)
    density = np.zeros(n_bins)

    for pos, diam in zip(positions, diameters):
        sigma = diam / 2.0
        # Periodic distance on circular track
        dx = bin_centers - pos
        dx = (dx + track_length / 2) % track_length - track_length / 2
        density += np.exp(-0.5 * (dx / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

    return density


# =============================================================================
# 4.  FNO surrogate rollout
# =============================================================================

def fno_rollout(
    positions: np.ndarray,    # (n,) candidate initial positions
    velocities: np.ndarray,   # (n,) candidate initial velocities
    diameters: np.ndarray,    # (n,) fixed particle diameters
    n_steps: int,
    fno: FourierNeuralOperator,
    norm_mean: np.ndarray,    # (n_bins,)
    norm_std: np.ndarray,     # (n_bins,)
    device: str,
    cfg: InverseFNOConfig,
) -> np.ndarray:
    """
    Build initial density from candidate positions, then roll out n_steps
    steps using the FNO+LSTM surrogate.

    The FNO expects a sliding window of k_history consecutive density frames.
    Since we only have one seed frame (computed from the candidate positions),
    we initialise the buffer by repeating that frame k times — effectively
    assuming zero velocity at t=0 from the network's perspective.  The LSTM
    hidden state is threaded through every step so that the recurrent
    momentum estimate accumulates over the rollout.

    Returns
    -------
    traj : (n_steps+1, n_bins) array in physical (unnormalised) units.
           traj[0] is the seed density, traj[i] is the prediction at step i.
    """
    k = fno.k_history

    # Compute seed density and normalise
    rho0      = positions_to_density(positions, cfg.track_length, cfg.n_density_bins, diameters)
    rho0_norm = (rho0 - norm_mean) / norm_std          # (n_bins,)

    # Build the initial rolling buffer: repeat seed frame k times → (k, n_bins)
    buffer = np.tile(rho0_norm[np.newaxis, :], (k, 1)).astype(np.float32)  # (k, n_bins)

    # Physical trajectory starts with the un-normalised seed
    traj_norm = [rho0_norm.copy()]    # list of (n_bins,) arrays
    lstm_state = None

    with torch.no_grad():
        for _ in range(n_steps):
            win = torch.from_numpy(buffer).unsqueeze(0).to(device)  # (1, k, n_bins)
            rho_next_t, lstm_state = fno(win, lstm_state)            # (1, n_bins), state
            rho_next = rho_next_t.squeeze(0).cpu().numpy()           # (n_bins,)
            traj_norm.append(rho_next)
            # Slide window: drop oldest, append newest
            buffer = np.concatenate([buffer[1:], rho_next[np.newaxis, :]], axis=0)  # (k, n_bins)

    traj_norm_arr = np.array(traj_norm)   # (n_steps+1, n_bins)
    # Denormalise
    traj = traj_norm_arr * norm_std[np.newaxis, :] + norm_mean[np.newaxis, :]
    return traj


# =============================================================================
# 5.  Physics simulation surrogate rollout
# =============================================================================

def physics_rollout(
    positions: np.ndarray,
    velocities: np.ndarray,
    diameters: np.ndarray,
    masses: np.ndarray,
    n_steps: int,
    cfg: InverseFNOConfig,
) -> np.ndarray:
    """
    Simulate forward using the physics engine and return the density trajectory.

    Returns
    -------
    traj : (n_steps+1, n_bins) array
    """
    system = ParticleSystem(cfg.track_length)
    for pos, vel, diam, mass in zip(positions, velocities, diameters, masses):
        system.add_particle(position=pos, velocity=vel, diameter=diam, mass=mass)

    duration = n_steps * cfg.save_interval
    states, _ = system.evolve(duration=duration, save_interval=cfg.save_interval)

    traj = []
    for state in states:
        pos_arr = state.positions
        diam_arr = state.diameters
        rho = positions_to_density(pos_arr, cfg.track_length, cfg.n_density_bins, diam_arr)
        traj.append(rho)

    # Pad if simulation ended early (e.g., all particles stopped)
    while len(traj) < n_steps + 1:
        traj.append(traj[-1].copy())

    return np.array(traj[: n_steps + 1])


# =============================================================================
# 6.  Objective function
# =============================================================================

class InverseObjective:
    """
    Encapsulates the optimisation objective for the FNO-in-the-loop inverse
    problem.

    The parameter vector θ is laid out as:
        θ = [x₀, x₁, …, x_{n-1},   v₀, v₁, …, v_{n-1}]
    all in [0, 1] (normalised), mapped back to physical units inside.
    """

    def __init__(
        self,
        observed_traj: np.ndarray,   # (T+1, n_bins) observed density sequence
        n_particles: int,
        diameters: np.ndarray,        # (n,) fixed true diameters
        masses: np.ndarray,           # (n,) fixed true masses
        fno: FourierNeuralOperator,
        norm_mean: np.ndarray,
        norm_std: np.ndarray,
        device: str,
        cfg: InverseFNOConfig,
        use_physics: bool = False,    # If True, use physics sim instead of FNO
    ):
        self.observed = observed_traj          # (T+1, n_bins)
        self.n_steps  = len(observed_traj) - 1
        self.n        = n_particles
        self.diameters = diameters
        self.masses   = masses
        self.fno      = fno
        self.norm_mean = norm_mean
        self.norm_std  = norm_std
        self.device   = device
        self.cfg      = cfg
        self.use_physics = use_physics

        self.track_length = cfg.track_length
        self.v_max = cfg.speed_range[1]
        self.n_evals = 0

    def decode(self, theta_01: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Map [0,1]^{2n} → physical (positions, velocities)."""
        pos = theta_01[:self.n] * self.track_length
        vel = (theta_01[self.n:] * 2 - 1) * self.v_max   # [-v_max, v_max]
        return pos, vel

    def __call__(self, theta_01: np.ndarray) -> float:
        """Return MSE loss between predicted and observed density trajectories."""
        self.n_evals += 1
        pos, vel = self.decode(np.clip(theta_01, 0.0, 1.0))

        try:
            if self.use_physics:
                pred_traj = physics_rollout(
                    pos, vel, self.diameters, self.masses,
                    self.n_steps, self.cfg,
                )
            else:
                pred_traj = fno_rollout(
                    pos, vel, self.diameters,
                    self.n_steps, self.fno,
                    self.norm_mean, self.norm_std,
                    self.device, self.cfg,
                )
        except Exception:
            return 1e6   # Penalise invalid configurations

        mse = float(np.mean((pred_traj - self.observed) ** 2))
        return mse


# =============================================================================
# 7.  Optimisation (CMA-ES or differential evolution)
# =============================================================================

def run_optimisation(
    objective: InverseObjective,
    rng: np.random.Generator,
    cfg: InverseFNOConfig,
) -> Tuple[np.ndarray, float, int]:
    """
    Run one restart of the optimiser.

    Returns
    -------
    theta_best : (2n,) best normalised parameter vector
    loss_best  : final objective value
    n_evals    : number of objective evaluations
    """
    dim = 2 * objective.n
    x0 = rng.uniform(0.0, 1.0, dim)

    bounds_01 = [(0.0, 1.0)] * dim

    if _HAS_CMA:
        opts = cma.CMAOptions()
        opts["maxiter"]  = cfg.cmaes_maxiter
        opts["bounds"]   = [[0.0] * dim, [1.0] * dim]
        opts["verbose"]  = -9   # suppress CMA output
        opts["seed"]     = int(rng.integers(0, 2**31))

        es = cma.CMAEvolutionStrategy(x0, cfg.cmaes_sigma0, opts)
        es.optimize(objective)
        result = es.result
        return result.xbest, float(result.fbest), objective.n_evals
    else:
        result = differential_evolution(
            objective,
            bounds=bounds_01,
            maxiter=cfg.diff_evo_maxiter,
            popsize=cfg.diff_evo_popsize,
            seed=int(rng.integers(0, 2**31)),
            tol=1e-6,
            mutation=(0.5, 1.5),
            recombination=0.9,
        )
        return result.x, float(result.fun), objective.n_evals


# =============================================================================
# 8.  Evaluation helpers
# =============================================================================

def compute_state_errors(
    theta_best: np.ndarray,
    true_positions: np.ndarray,    # (n,) sorted
    true_velocities: np.ndarray,   # (n,) matching ordering
    n_particles: int,
    objective: InverseObjective,
) -> Dict[str, float]:
    """
    Compute position and velocity RMSE between best estimate and ground truth.

    We try all circular permutations of inferred positions to handle the
    label-permutation degeneracy (single-file particles are ordered on the
    track, so we match by sorted order).
    """
    pos_est, vel_est = objective.decode(np.clip(theta_best, 0.0, 1.0))

    # Sort both by position (single-file ordering is canonical)
    sort_est  = np.argsort(pos_est)
    sort_true = np.argsort(true_positions)

    pos_est_sorted  = pos_est[sort_est]
    vel_est_sorted  = vel_est[sort_est]
    true_pos_sorted = true_positions[sort_true]
    true_vel_sorted = true_velocities[sort_true]

    # Circular position error (minimum over wrap-arounds)
    diff_pos = pos_est_sorted - true_pos_sorted
    diff_pos = (diff_pos + objective.track_length / 2) % objective.track_length - objective.track_length / 2
    pos_rmse = float(np.sqrt(np.mean(diff_pos ** 2)))

    vel_rmse = float(np.sqrt(np.mean((vel_est_sorted - true_vel_sorted) ** 2)))

    return {
        "pos_rmse": pos_rmse,
        "vel_rmse": vel_rmse,
        "pos_error_mean": float(np.mean(np.abs(diff_pos))),
        "vel_error_mean": float(np.mean(np.abs(vel_est_sorted - true_vel_sorted))),
    }


# =============================================================================
# 9.  Multi-trial inverse loop over a set of test systems
# =============================================================================

def run_inverse_trials(
    test_trajectories: List[Dict],
    fno: FourierNeuralOperator,
    norm_mean: np.ndarray,
    norm_std: np.ndarray,
    horizon_steps: List[int],
    device: str,
    cfg: InverseFNOConfig,
    rng: np.random.Generator,
    use_physics: bool = False,
    label: str = "FNO",
) -> List[Dict]:
    """
    For each (test system, horizon) pair run `cfg.n_trials` CMA-ES restarts
    and record errors.

    Returns
    -------
    records : list of per-trial result dicts
    """
    records = []
    surrogate = "Physics" if use_physics else label

    for sys_idx, traj in enumerate(test_trajectories):
        n_particles = traj["n_particles"]
        true_velocities = np.array(traj["initial_velocities"])
        true_diameters  = np.array(traj["diameters"])
        true_masses     = np.array(traj["masses"])
        density_fields  = np.array(traj["density_fields"])   # (T_full, n_bins)

        # Recover initial positions — stored from experiment_1 if dataset was
        # generated after the initial_positions patch; otherwise reconstruct
        # using the evenly-spaced placement (same heuristic as experiment_4).
        if "initial_positions" in traj:
            true_positions = np.array(traj["initial_positions"])
        else:
            warnings.warn(
                f"System {sys_idx}: 'initial_positions' not in dataset; "
                "using evenly-spaced reconstruction as ground truth. "
                "Re-generate the dataset for accurate position error metrics.",
                UserWarning,
            )
            spacing = cfg.track_length / n_particles
            true_positions = np.array([spacing * i + spacing * 0.5 for i in range(n_particles)])

        print(f"\n  [{surrogate}] System {sys_idx+1}/{len(test_trajectories)}, "
              f"n={n_particles}, T_full={len(density_fields)}")

        for T_steps in horizon_steps:
            if T_steps >= len(density_fields):
                T_steps = len(density_fields) - 1

            observed = density_fields[:T_steps + 1]   # (T_steps+1, n_bins)

            for trial in range(cfg.n_trials):
                t_start = time.time()
                objective = InverseObjective(
                    observed_traj=observed,
                    n_particles=n_particles,
                    diameters=true_diameters,
                    masses=true_masses,
                    fno=fno,
                    norm_mean=norm_mean,
                    norm_std=norm_std,
                    device=device,
                    cfg=cfg,
                    use_physics=use_physics,
                )

                theta_best, loss_best, n_evals = run_optimisation(objective, rng, cfg)
                elapsed = time.time() - t_start

                errors = compute_state_errors(
                    theta_best, true_positions, true_velocities,
                    n_particles, objective,
                )

                rec = {
                    "surrogate":    surrogate,
                    "system_idx":   sys_idx,
                    "n_particles":  n_particles,
                    "T_steps":      T_steps,
                    "trial":        trial,
                    "loss_final":   loss_best,
                    "n_evals":      n_evals,
                    "elapsed_s":    elapsed,
                    **errors,
                }
                records.append(rec)

                print(f"    T={T_steps:4d} trial={trial} | "
                      f"pos_rmse={errors['pos_rmse']:.2f}  "
                      f"vel_rmse={errors['vel_rmse']:.2f}  "
                      f"loss={loss_best:.5f}  ({elapsed:.1f}s)")

    return records


# =============================================================================
# 10.  Cramér-Rao Bound analysis
# =============================================================================

def compute_fim(
    objective: InverseObjective,
    theta_true_01: np.ndarray,   # Ground-truth θ in [0,1]
    sigma2: float,
    eps: float,
) -> np.ndarray:
    """
    Numerically compute the Fisher Information Matrix via finite differences.

    For Gaussian likelihood  p(ρ_obs | θ) ∝ exp(−‖ρ_fno(θ) − ρ_obs‖² / 2σ²),
    the FIM is:

        I(θ)ᵢⱼ = (1/σ²) · (∂ρ/∂θᵢ)ᵀ · (∂ρ/∂θⱼ)

    where the Jacobian ∂ρ/∂θ ∈ ℝ^{(T·B) × 2n} is estimated by finite differences.

    Returns
    -------
    FIM : (2n, 2n) symmetric positive semi-definite matrix
    """
    dim = len(theta_true_01)
    n_bins = objective.cfg.n_density_bins
    T = objective.n_steps

    theta0 = np.clip(theta_true_01.copy(), eps * 2, 1 - eps * 2)

    # Evaluate rollout at θ₀
    pos0, vel0 = objective.decode(theta0)
    if objective.use_physics:
        rollout0 = physics_rollout(
            pos0, vel0, objective.diameters, objective.masses,
            T, objective.cfg,
        )
    else:
        rollout0 = fno_rollout(
            pos0, vel0, objective.diameters,
            T, objective.fno, objective.norm_mean, objective.norm_std,
            objective.device, objective.cfg,
        )
    f0 = rollout0.flatten()  # (T+1)*n_bins

    # Build Jacobian column by column
    J = np.zeros((len(f0), dim))  # ((T+1)*n_bins, 2n)

    for k in range(dim):
        theta_plus = theta0.copy()
        theta_plus[k] = min(theta0[k] + eps, 1.0)
        theta_minus = theta0.copy()
        theta_minus[k] = max(theta0[k] - eps, 0.0)

        pos_plus,  vel_plus  = objective.decode(theta_plus)
        pos_minus, vel_minus = objective.decode(theta_minus)

        actual_eps = (theta_plus[k] - theta_minus[k])

        if objective.use_physics:
            f_plus  = physics_rollout(pos_plus, vel_plus, objective.diameters, objective.masses, T, objective.cfg).flatten()
            f_minus = physics_rollout(pos_minus, vel_minus, objective.diameters, objective.masses, T, objective.cfg).flatten()
        else:
            f_plus  = fno_rollout(pos_plus, vel_plus, objective.diameters, T, objective.fno, objective.norm_mean, objective.norm_std, objective.device, objective.cfg).flatten()
            f_minus = fno_rollout(pos_minus, vel_minus, objective.diameters, T, objective.fno, objective.norm_mean, objective.norm_std, objective.device, objective.cfg).flatten()

        # Decode physical step size for dimensionally correct Jacobian
        if k < objective.n:
            phys_eps = actual_eps * objective.track_length
        else:
            phys_eps = actual_eps * 2 * objective.cfg.speed_range[1]

        J[:, k] = (f_plus - f_minus) / (phys_eps + 1e-12)

    # FIM = J^T J / σ²
    fim = (J.T @ J) / sigma2
    return fim


def crb_analysis(
    test_trajectories: List[Dict],
    fno: FourierNeuralOperator,
    norm_mean: np.ndarray,
    norm_std: np.ndarray,
    horizon_steps: List[int],
    sigma2: float,
    device: str,
    cfg: InverseFNOConfig,
    rng: np.random.Generator,
) -> Dict:
    """
    Compute the Cramér-Rao Bound as a function of the observation horizon T.

    Also estimates the Lyapunov exponent and shows the theoretical
    FIM ~ exp(2λT) scaling.

    Returns
    -------
    crb_results : dict with per-horizon CRB values and Lyapunov info
    """
    print("\n" + "=" * 60)
    print("CRB ANALYSIS")
    print("=" * 60)

    crb_pos_per_horizon   = []   # CRB std on position per horizon
    crb_vel_per_horizon   = []   # CRB std on velocity per horizon
    fim_trace_per_horizon = []
    fim_mineig_per_horizon = []
    lyapunov_times = []

    track_length = cfg.track_length

    # Use a small subset for speed
    systems = test_trajectories[:cfg.crb_n_systems]

    for sys_idx, traj in enumerate(systems):
        n_particles     = traj["n_particles"]
        true_velocities = np.array(traj["initial_velocities"])
        true_diameters  = np.array(traj["diameters"])
        true_masses     = np.array(traj["masses"])
        density_fields  = np.array(traj["density_fields"])

        if "initial_positions" in traj:
            true_positions = np.array(traj["initial_positions"])
        else:
            spacing = cfg.track_length / n_particles
            true_positions = np.array([spacing * i + spacing * 0.5 for i in range(n_particles)])

        print(f"\n  System {sys_idx+1}/{len(systems)}, n={n_particles}")

        # --- Lyapunov exponent -----------------------------------------------------------
        system = ParticleSystem(cfg.track_length)
        for pos, vel, diam, mass in zip(true_positions, true_velocities, true_diameters, true_masses):
            system.add_particle(position=pos, velocity=vel, diameter=diam, mass=mass)
        lyap = system.compute_lyapunov_exponent(perturbation=1e-8, duration=min(50.0, 0.1 * traj.get("simulation_time", 500.0)))
        t_lyap = 1.0 / max(lyap, 1e-6)
        lyapunov_times.append(t_lyap)
        print(f"    Lyapunov exponent λ = {lyap:.4f},  T_Lyap = {t_lyap:.2f} s")

        # True θ in [0,1]
        theta_true_01 = np.concatenate([
            true_positions / track_length,
            (true_velocities / cfg.speed_range[1] + 1) / 2,
        ])
        theta_true_01 = np.clip(theta_true_01, 0.01, 0.99)

        sys_crb_pos = []
        sys_crb_vel = []
        sys_fim_trace = []
        sys_fim_mineig = []

        for T_steps in horizon_steps:
            if T_steps >= len(density_fields):
                T_steps = len(density_fields) - 1

            observed = density_fields[:T_steps + 1]

            objective = InverseObjective(
                observed_traj=observed,
                n_particles=n_particles,
                diameters=true_diameters,
                masses=true_masses,
                fno=fno,
                norm_mean=norm_mean,
                norm_std=norm_std,
                device=device,
                cfg=cfg,
                use_physics=False,   # FNO for speed
            )

            try:
                fim = compute_fim(objective, theta_true_01, sigma2, cfg.fim_eps)

                # Regularise before inversion
                lam_reg = 1e-6 * np.trace(fim) / max(fim.shape[0], 1)
                crb_matrix = np.linalg.inv(fim + lam_reg * np.eye(fim.shape[0]))

                # CRB on position indices [0:n], velocity indices [n:2n]
                # The FIM was built in physical units (metres, m/s)
                crb_pos_variance = np.mean(np.diag(crb_matrix)[:n_particles])
                crb_vel_variance = np.mean(np.diag(crb_matrix)[n_particles:])

                crb_pos_std = float(np.sqrt(max(crb_pos_variance, 0)))
                crb_vel_std = float(np.sqrt(max(crb_vel_variance, 0)))

                eigenvalues = np.linalg.eigvalsh(fim)
                fim_trace  = float(np.trace(fim))
                fim_mineig = float(eigenvalues[0])

            except np.linalg.LinAlgError:
                crb_pos_std = np.nan
                crb_vel_std = np.nan
                fim_trace   = np.nan
                fim_mineig  = np.nan

            sys_crb_pos.append(crb_pos_std)
            sys_crb_vel.append(crb_vel_std)
            sys_fim_trace.append(fim_trace)
            sys_fim_mineig.append(fim_mineig)

            print(f"    T={T_steps:4d} | CRB_pos={crb_pos_std:.3f}  CRB_vel={crb_vel_std:.3f}  "
                  f"FIM_trace={fim_trace:.2e}")

        crb_pos_per_horizon.append(sys_crb_pos)
        crb_vel_per_horizon.append(sys_crb_vel)
        fim_trace_per_horizon.append(sys_fim_trace)
        fim_mineig_per_horizon.append(sys_fim_mineig)

    # Average over systems
    def safe_mean(arr2d):
        a = np.array(arr2d, dtype=float)
        return np.nanmean(a, axis=0).tolist()

    mean_t_lyap = float(np.mean(lyapunov_times))

    # Fit FIM ~ A * exp(2λT) to the trace data (log-linear regression)
    t_phys = np.array(horizon_steps) * cfg.save_interval   # seconds
    mean_fim_trace = np.array(safe_mean(fim_trace_per_horizon))
    lyap_estimate_from_fim = np.nan

    valid = np.isfinite(mean_fim_trace) & (mean_fim_trace > 0)
    if valid.sum() >= 3:
        log_trace = np.log(mean_fim_trace[valid])
        t_valid   = t_phys[valid]
        # Linear regression: log(trace) = a + 2λ·t
        coeffs = np.polyfit(t_valid, log_trace, 1)
        lyap_estimate_from_fim = float(coeffs[0] / 2)   # slope / 2 = λ
        print(f"\n  FIM-based Lyapunov estimate: λ_FIM = {lyap_estimate_from_fim:.4f} s⁻¹")
        print(f"  (Mean direct estimate:       λ_dir = {1/mean_t_lyap:.4f} s⁻¹)")

    # T* heuristic: first T where CRB_pos < 0.1 × track_length / n
    mean_crb_pos = safe_mean(crb_pos_per_horizon)
    typical_particle_spacing = cfg.track_length / (np.mean([t["n_particles"] for t in systems]) + 1)
    threshold_pos = 0.1 * typical_particle_spacing
    t_star_steps = None
    for k, (T_s, crb_p) in enumerate(zip(horizon_steps, mean_crb_pos)):
        if np.isfinite(crb_p) and crb_p < threshold_pos:
            t_star_steps = T_s
            break

    if t_star_steps is not None:
        t_star_s = t_star_steps * cfg.save_interval
        ratio = t_star_s / mean_t_lyap
        print(f"\n  T* (CRB_pos < {threshold_pos:.1f}) = {t_star_steps} steps = {t_star_s:.2f} s")
        print(f"  T* / T_Lyap = {ratio:.2f}  (expected ~1–3 for well-conditioned inference)")
    else:
        t_star_steps = None
        print("\n  T* not reached within the tested horizon range.")

    return {
        "horizons_steps": horizon_steps,
        "horizons_s": (np.array(horizon_steps) * cfg.save_interval).tolist(),
        "mean_crb_pos": mean_crb_pos,
        "mean_crb_vel": safe_mean(crb_vel_per_horizon),
        "mean_fim_trace": mean_fim_trace.tolist(),
        "mean_fim_mineig": safe_mean(fim_mineig_per_horizon),
        "lyapunov_times_s": lyapunov_times,
        "mean_lyapunov_time_s": mean_t_lyap,
        "lyap_estimate_from_fim": float(lyap_estimate_from_fim) if not np.isnan(lyap_estimate_from_fim) else None,
        "t_star_steps": t_star_steps,
        "t_star_s": (t_star_steps * cfg.save_interval) if t_star_steps else None,
        "t_star_over_t_lyap": (t_star_steps * cfg.save_interval / mean_t_lyap) if t_star_steps else None,
    }


# =============================================================================
# 11.  Main experiment function
# =============================================================================

def experiment_7b_inverse_fno(
    config: "ExperimentConfig",   # from paper1_experiments.py
    dataset: Dict,
    smoke_test: bool = False,
) -> Dict:
    """
    Main entry point matching the calling convention of the other paper1 experiments.

    Parameters
    ----------
    config  : ExperimentConfig (from paper1_experiments.py)
    dataset : Dict with keys 'train', 'val', 'test', each a list of trajectory dicts
    smoke_test : If True, run a tiny version for debugging

    Returns
    -------
    results : nested dict ready to be JSON-serialised
    """
    print("=" * 60)
    print("EXPERIMENT 7b: FNO-In-The-Loop Inverse Problem (Framing A)")
    print("=" * 60)

    # ── derive InverseFNOConfig from ExperimentConfig ─────────────────────────
    cfg = InverseFNOConfig(
        output_dir=config.output_dir,
        track_length=config.track_length,
        n_density_bins=config.n_density_bins,
        save_interval=config.save_interval,
        fno_modes=config.fno_modes,
        fno_hidden_channels=config.fno_hidden_channels,
        fno_layers=config.fno_layers,
        k_history=config.k_history,
        lstm_hidden=config.lstm_hidden,
        lstm_layers=config.lstm_layers,
        speed_range=config.speed_range,
    )
    if smoke_test:
        cfg = smoke_test_config(cfg)

    rng = np.random.default_rng(config.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    results: Dict = {
        "fno_inverse":     [],
        "physics_inverse": [],
        "crb_analysis":    {},
        "summary":         {},
    }

    # ── 1. Load FNO ────────────────────────────────────────────────────────────
    print("\nLoading trained FNO from Experiment 3...")
    fno, val_loss = load_fno(cfg, device)
    sigma2 = cfg.observation_sigma2 if cfg.observation_sigma2 is not None else val_loss
    print(f"  σ² (noise floor for FIM): {sigma2:.6f}")

    # ── 2. Build normalisation stats ───────────────────────────────────────────
    print("\nBuilding normalisation statistics from training split...")
    norm_mean, norm_std = build_normalisation_stats(dataset, cfg)

    # ── 3. Select test trajectories ────────────────────────────────────────────
    test_trajs = dataset["test"][: cfg.n_test_systems]
    print(f"\nUsing {len(test_trajs)} test systems.")

    # ── 4. Horizon sweep ───────────────────────────────────────────────────────
    if cfg.horizon_steps is None:
        max_T = min(len(test_trajs[0]["density_fields"]) - 1, 500)
        cfg.horizon_steps = [1, 2, 5, 10, 20, 50, 100, 200, 500]
        cfg.horizon_steps = [T for T in cfg.horizon_steps if T <= max_T]
    print(f"\nHorizon sweep: {cfg.horizon_steps} steps "
          f"({[T * cfg.save_interval for T in cfg.horizon_steps]} s)")

    # ── 5. FNO-surrogate inverse ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SECTION A: FNO Surrogate Inverse")
    print("=" * 60)
    fno_records = run_inverse_trials(
        test_trajs, fno, norm_mean, norm_std,
        cfg.horizon_steps, device, cfg, rng,
        use_physics=False, label="FNO",
    )
    results["fno_inverse"] = fno_records

    # ── 6. Physics-sim inverse (ground truth baseline) ─────────────────────────
    print("\n" + "=" * 60)
    print("SECTION B: Physics Simulation Inverse (Ground Truth Baseline)")
    print("=" * 60)
    # Limit to shorter horizons for physics (expensive)
    short_horizons = [T for T in cfg.horizon_steps if T <= 50]
    if short_horizons:
        physics_records = run_inverse_trials(
            test_trajs[:min(5, len(test_trajs))],
            fno, norm_mean, norm_std,
            short_horizons, device, cfg, rng,
            use_physics=True, label="Physics",
        )
        results["physics_inverse"] = physics_records
    else:
        print("  Skipping (no short horizons in sweep).")

    # ── 7. Cramér-Rao Bound analysis ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SECTION C: Cramér-Rao Bound Analysis")
    print("=" * 60)
    crb_res = crb_analysis(
        test_trajs, fno, norm_mean, norm_std,
        cfg.horizon_steps, sigma2, device, cfg, rng,
    )
    results["crb_analysis"] = crb_res

    # ── 8. Compile summary statistics ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    summary = _compile_summary(fno_records, crb_res, cfg)
    results["summary"] = summary

    _print_summary(summary, crb_res)

    # ── 9. Save results ────────────────────────────────────────────────────────
    out_path = os.path.join(cfg.output_dir, "experiment_7b_results.json")
    os.makedirs(cfg.output_dir, exist_ok=True)
    try:
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=_json_safe)
        print(f"\nResults saved → {out_path}")
    except Exception as e:
        print(f"\nWarning: could not save results: {e}")

    return results


# =============================================================================
# 12.  Summary helpers
# =============================================================================

def _compile_summary(
    fno_records: List[Dict],
    crb_res: Dict,
    cfg: InverseFNOConfig,
) -> Dict:
    """Build a compact summary dict from the raw per-trial records."""

    def records_by_horizon(records, T):
        return [r for r in records if r["T_steps"] == T]

    horizon_summary = []
    for T in cfg.horizon_steps:
        recs = records_by_horizon(fno_records, T)
        if not recs:
            continue
        pos_rmses = [r["pos_rmse"] for r in recs]
        vel_rmses = [r["vel_rmse"] for r in recs]
        horizon_summary.append({
            "T_steps": T,
            "T_s": T * cfg.save_interval,
            "n_trials": len(recs),
            "pos_rmse_mean": float(np.mean(pos_rmses)),
            "pos_rmse_std":  float(np.std(pos_rmses)),
            "vel_rmse_mean": float(np.mean(vel_rmses)),
            "vel_rmse_std":  float(np.std(vel_rmses)),
            "loss_mean":     float(np.mean([r["loss_final"] for r in recs])),
        })

    return {
        "method": "FNO-in-the-loop CMA-ES",
        "n_test_systems": cfg.n_test_systems,
        "n_trials_per_system": cfg.n_trials,
        "optimizer": "CMA-ES" if _HAS_CMA else "DifferentialEvolution",
        "horizon_summary": horizon_summary,
        "mean_lyapunov_time_s": crb_res.get("mean_lyapunov_time_s"),
        "t_star_steps": crb_res.get("t_star_steps"),
        "t_star_s": crb_res.get("t_star_s"),
        "t_star_over_t_lyap": crb_res.get("t_star_over_t_lyap"),
    }


def _print_summary(summary: Dict, crb_res: Dict) -> None:
    print(f"\n  Method : {summary['method']}")
    print(f"  Optimizer: {summary['optimizer']}")
    print(f"  Mean T_Lyap: {summary['mean_lyapunov_time_s']:.2f} s")
    if summary["t_star_steps"]:
        print(f"  T* (well-conditioned inference): "
              f"{summary['t_star_steps']} steps = {summary['t_star_s']:.2f} s "
              f"({summary['t_star_over_t_lyap']:.2f} × T_Lyap)")
    print()
    print(f"  {'T_steps':>8}  {'T(s)':>6}  {'pos_rmse':>10}  {'vel_rmse':>10}")
    print("  " + "-" * 42)
    for row in summary["horizon_summary"]:
        print(f"  {row['T_steps']:>8}  {row['T_s']:>6.1f}  "
              f"{row['pos_rmse_mean']:>8.2f}±{row['pos_rmse_std']:<5.2f}  "
              f"{row['vel_rmse_mean']:>8.2f}±{row['vel_rmse_std']:<5.2f}")


def _json_safe(obj):
    """JSON serialiser for numpy types."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")


# =============================================================================
# 13.  Standalone entry point
# =============================================================================

def _standalone_main():
    """
    Run Experiment 7b standalone (without paper1_experiments.py driver).

    Generates a small synthetic dataset, trains or loads FNO, and runs the
    full inverse experiment.
    """
    parser = argparse.ArgumentParser(description="Experiment 7b: FNO Inverse Problem")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run a tiny version for debugging (< 60 s)")
    parser.add_argument("--output-dir", default="./results/paper1",
                        help="Directory containing fno_checkpoint.pt and for output")
    parser.add_argument("--n-test", type=int, default=10,
                        help="Number of test systems to invert")
    parser.add_argument("--n-trials", type=int, default=5,
                        help="CMA-ES restarts per (system, horizon)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Build a minimal ExperimentConfig-compatible object
    from paper1_experiments import ExperimentConfig, experiment_1_generate_dataset

    config = ExperimentConfig(
        experiment_name="7b_standalone",
        seed=args.seed,
        output_dir=args.output_dir,
        n_training_trajectories=50,
        n_validation_trajectories=10,
        n_test_trajectories=args.n_test,
        simulation_time=50.0,   # shorter for standalone
    )

    # Generate or load dataset
    dataset_path = os.path.join(args.output_dir, "dataset.json")
    if os.path.exists(dataset_path):
        print(f"Loading existing dataset from {dataset_path}...")
        with open(dataset_path) as f:
            dataset = json.load(f)
    else:
        print("Generating dataset (experiment_1)...")
        dataset = experiment_1_generate_dataset(config)

    results = experiment_7b_inverse_fno(config, dataset, smoke_test=args.smoke_test)
    print("\nDone.")


if __name__ == "__main__":
    _standalone_main()
