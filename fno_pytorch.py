"""
Neural Operators for Impulsive Dynamical Systems - PyTorch Implementation

This module implements the machine learning methodology for Paper 1:
"Neural Operators for Density Evolution in Impulsive Dynamical Systems"

Ported from NumPy to PyTorch for GPU acceleration and automatic differentiation.

Core idea:
- The Perron-Frobenius (PF) operator governs density evolution
- PF is LINEAR even when particle dynamics are nonlinear/chaotic
- Neural operators can learn this mapping: ρ(t) → ρ(t + Δt)
- Once learned, we can predict density beyond the Lyapunov horizon

Key challenge:
- Hard-sphere collisions are discontinuous (velocity jumps)
- Standard PINNs assume smooth PDEs
- We handle this by:
  1. Working with coarse-grained densities (smooths discontinuities)
  2. Learning the operator directly from data (no PDE residual loss)
  3. Using Fourier Neural Operators (naturally handle periodic BC)

Architecture options:
1. Fourier Neural Operator (FNO) - best for periodic domains
2. DeepONet - good for general operator learning
3. Transformer-based - for very long sequences

Author: Kiran Bhattacharyya
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple, Optional, Dict, Callable
from dataclasses import dataclass, field
import json


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class DensityTrajectory:
    """
    A trajectory of density fields over time.

    This is the training data format:
    - Input: a sliding window of k consecutive density snapshots
    - Target: density at time t + delta_t * n_steps_ahead

    For supervised learning of the Perron-Frobenius operator.
    Stacking k frames gives the network finite-difference velocity
    information that a single snapshot cannot provide.
    """
    time_points: np.ndarray   # Shape: (T,)
    density_fields: np.ndarray  # Shape: (T, n_bins)
    delta_t: float              # Time step between consecutive snapshots

    # Metadata
    system_params: Dict = field(default_factory=dict)

    @property
    def n_timesteps(self) -> int:
        return len(self.time_points)

    @property
    def n_bins(self) -> int:
        return self.density_fields.shape[1]

    def get_input_output_pairs(
        self,
        n_steps_ahead: int = 1,
        k_history: int = 4
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create (input, target) pairs for training with stacked history windows.

        Each input sample is a stack of k consecutive frames ending at time t;
        the target is the frame at time t + n_steps_ahead.

        Valid sample indices i satisfy:
            i in [0, T - k_history - n_steps_ahead)

        Args:
            n_steps_ahead: How many steps ahead to predict (default 1)
            k_history    : Number of past frames to stack as input (default 4)

        Returns:
            inputs  : Shape (n_samples, k_history, n_bins)
            targets : Shape (n_samples, n_bins)
        """
        n_samples = self.n_timesteps - (k_history - 1) - n_steps_ahead
        if n_samples <= 0:
            raise ValueError(
                f"Trajectory has {self.n_timesteps} timesteps but k_history={k_history} "
                f"and n_steps_ahead={n_steps_ahead} require at least "
                f"{k_history + n_steps_ahead} timesteps."
            )
        inputs = np.stack(
            [self.density_fields[i: i + k_history] for i in range(n_samples)],
            axis=0
        )  # (n_samples, k_history, n_bins)
        targets = self.density_fields[
            k_history - 1 + n_steps_ahead:
            k_history - 1 + n_steps_ahead + n_samples
        ]  # (n_samples, n_bins)
        return inputs, targets


@dataclass
class CollisionSequence:
    """
    Sequence of collision events for inverse problem (Pillar 1).
    
    The challenge: infer particle masses and diameters from this
    sequence alone (the "shadow" inverse problem).
    """
    times: np.ndarray  # Collision timestamps
    particle_pairs: np.ndarray  # Shape: (n_collisions, 2) - which particles
    momentum_transfers: np.ndarray  # |Δp| at each collision
    positions: np.ndarray  # Where collisions occurred
    
    # Ground truth (hidden during inference)
    true_masses: Optional[np.ndarray] = None
    true_diameters: Optional[np.ndarray] = None
    
    @property
    def n_collisions(self) -> int:
        return len(self.times)
    
    def get_inter_collision_times(self) -> np.ndarray:
        """Time between successive collisions."""
        return np.diff(self.times)
    
    def get_collision_rate(self) -> float:
        """Average collision rate."""
        if self.n_collisions < 2:
            return 0.0
        total_time = self.times[-1] - self.times[0]
        return self.n_collisions / total_time


# =============================================================================
# PyTorch Dataset
# =============================================================================

class DensityDataset(Dataset):
    """
    PyTorch Dataset for density trajectories.

    Each sample:
        input  : (k_history, n_bins) — stacked consecutive density frames
        target : (n_bins,)           — density at the next step

    The stacked history lets the model infer velocity (1st diff) and
    acceleration (2nd diff) from finite differences between frames.
    """

    def __init__(
        self,
        trajectories: List[DensityTrajectory],
        n_steps_ahead: int = 1,
        normalize: bool = True,
        k_history: int = 4
    ):
        self.trajectories  = trajectories
        self.n_steps_ahead = n_steps_ahead
        self.normalize     = normalize
        self.k_history     = k_history

        # Collect all (window, target) pairs
        all_inputs:  List[np.ndarray] = []
        all_targets: List[np.ndarray] = []

        for traj in trajectories:
            inputs, targets = traj.get_input_output_pairs(n_steps_ahead, k_history)
            all_inputs.append(inputs)
            all_targets.append(targets)

        self.inputs  = np.concatenate(all_inputs,  axis=0)  # (N, k, n_bins)
        self.targets = np.concatenate(all_targets, axis=0)  # (N, n_bins)

        # Normalization statistics
        # input_mean/std: per-bin averages across all samples and all k frames
        # shape (1, n_bins) so they broadcast against (k, n_bins) in __getitem__
        if self.normalize:
            self.input_mean  = self.inputs.mean(axis=(0, 1), keepdims=True).squeeze(0)   # (1, n_bins)
            self.input_std   = self.inputs.std( axis=(0, 1), keepdims=True).squeeze(0) + 1e-8
            self.target_mean = self.targets.mean(axis=0)   # (n_bins,)  — no keepdims so __getitem__ returns 1D
            self.target_std  = self.targets.std( axis=0) + 1e-8
        else:
            self.input_mean  = 0.0
            self.input_std   = 1.0
            self.target_mean = 0.0
            self.target_std  = 1.0

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # inputs[idx]: (k, n_bins);  targets[idx]: (n_bins,)
        input_window  = (self.inputs[idx]  - self.input_mean)  / self.input_std   # (k, n_bins)
        target_density = (self.targets[idx] - self.target_mean) / self.target_std  # (n_bins,)
        return (
            torch.from_numpy(input_window.astype(np.float32)),
            torch.from_numpy(target_density.astype(np.float32))
        )

    def denormalize_output(self, normalized_output: torch.Tensor) -> torch.Tensor:
        """Convert normalized predictions back to physical space."""
        if isinstance(self.target_std, np.ndarray):
            std  = torch.from_numpy(self.target_std.astype(np.float32)).to(normalized_output.device)
            mean = torch.from_numpy(self.target_mean.astype(np.float32)).to(normalized_output.device)
        else:
            std, mean = self.target_std, self.target_mean
        return normalized_output * std + mean


# =============================================================================
# Fourier Neural Operator Components
# =============================================================================

class SpectralConvolution1d(nn.Module):
    """
    Spectral convolution layer for FNO.
    
    Operates in Fourier space:
    1. FFT the input
    2. Multiply by learnable weights (in frequency domain)
    3. IFFT back to physical space
    
    This naturally handles periodic boundary conditions
    (perfect for circular track).
    
    Paper reference: Li et al. (2020) "Fourier Neural Operator"
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes: int,  # Number of Fourier modes to keep
    ):
        super(SpectralConvolution1d, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_modes = n_modes
        
        # Learnable weights in Fourier space (complex-valued)
        # Shape: (in_channels, out_channels, n_modes)
        scale = 1.0 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            torch.randn(in_channels, out_channels, n_modes, 2) * scale
        )
    
    def complex_multiply(self, input_fft, weights):
        """
        Complex multiplication in Fourier space.
        
        Args:
            input_fft: (batch, in_channels, modes) complex tensor
            weights: (in_channels, out_channels, modes, 2) real tensor
            
        Returns:
            (batch, out_channels, modes) complex tensor
        """
        # Convert weights to complex
        weights_complex = torch.view_as_complex(weights)  # (in_ch, out_ch, modes)
        
        # Einstein summation for batched complex multiplication
        # (batch, in_ch, modes) x (in_ch, out_ch, modes) -> (batch, out_ch, modes)
        output = torch.einsum('bim,iom->bom', input_fft, weights_complex)
        
        return output
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply spectral convolution.
        
        Args:
            x: Input tensor, shape (batch, in_channels, n_spatial)
            
        Returns:
            Output tensor, shape (batch, out_channels, n_spatial)
        """
        batch_size = x.shape[0]
        n_spatial = x.shape[-1]
        
        # FFT along spatial dimension (real FFT)
        x_ft = torch.fft.rfft(x, dim=-1)  # (batch, in_channels, n_spatial//2+1)
        
        # Keep only first n_modes
        x_ft_modes = x_ft[..., :self.n_modes]
        
        # Multiply by complex weights
        out_ft = self.complex_multiply(x_ft_modes, self.weights)
        
        # Pad with zeros for higher modes
        out_ft_padded = torch.zeros(
            batch_size, self.out_channels, n_spatial // 2 + 1,
            dtype=torch.cfloat,
            device=x.device
        )
        out_ft_padded[..., :self.n_modes] = out_ft
        
        # IFFT back to physical space
        out = torch.fft.irfft(out_ft_padded, n=n_spatial, dim=-1)
        
        return out


class FourierLayer(nn.Module):
    """
    One layer of the Fourier Neural Operator.
    
    Combines:
    1. Spectral convolution (global, in Fourier space)
    2. Local linear transform (pointwise, 1x1 conv)
    3. Nonlinearity (GeLU)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes: int,
    ):
        super(FourierLayer, self).__init__()
        
        self.spectral_conv = SpectralConvolution1d(in_channels, out_channels, n_modes)
        
        # Local linear transform (1x1 convolution)
        self.local_transform = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        
        self.activation = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Shape (batch, in_channels, n_spatial)
            
        Returns:
            Shape (batch, out_channels, n_spatial)
        """
        # Spectral path
        x_spectral = self.spectral_conv(x)
        
        # Local path
        x_local = self.local_transform(x)
        
        # Combine and apply nonlinearity
        out = self.activation(x_spectral + x_local)
        
        return out


class FourierNeuralOperator(nn.Module):
    """
    Fourier Neural Operator (1D) with stacked history input and LSTM recurrence.

    Architecture
    ------------
    1. **Lift**   : Conv1d(k_history, hidden_channels, kernel_size=1)
                    — treats the k stacked density frames as input channels
    2. **FNO trunk** : N × FourierLayer(hidden, hidden, n_modes)
    3. **LSTM**   : processes (batch, n_bins, hidden) treating spatial bins
                    as sequence steps → captures cross-position correlations
                    and maintains momentum state across autoregressive rollout
    4. **Project**: Linear(lstm_hidden, 1) + ReLU → (batch, n_bins)

    Input  : (batch, k_history, n_bins)  — k consecutive density windows
    Output : (batch, n_bins),            — predicted density at t + Δt
             plus LSTM hidden state  (h_n, c_n)  for autoregressive threading

    The model learns the Perron-Frobenius map
        [ρ(t-k+1), …, ρ(t)] → ρ(t + Δt)
    effectively approximating the operator while encoding momentum
    structure implicitly through finite differences between frames.
    """

    def __init__(
        self,
        n_bins:          int   = 100,
        n_modes:         int   = 16,
        hidden_channels: int   = 32,
        n_layers:        int   = 4,
        delta_t:         float = 1.0,
        k_history:       int   = 4,
        lstm_hidden:     int   = 64,
        lstm_layers:     int   = 1,
    ):
        super(FourierNeuralOperator, self).__init__()

        self.n_bins          = n_bins
        self.n_modes         = n_modes
        self.hidden_channels = hidden_channels
        self.n_layers        = n_layers
        self.delta_t         = delta_t
        self.k_history       = k_history
        self.lstm_hidden     = lstm_hidden
        self.lstm_layers     = lstm_layers

        # Lifting layer: k_history channels → hidden_channels
        # Input treated as (batch, k_history, n_bins) — k is the channel dim
        self.lift = nn.Conv1d(k_history, hidden_channels, kernel_size=1)

        # Fourier layers (unchanged spatial structure)
        self.fourier_layers = nn.ModuleList([
            FourierLayer(hidden_channels, hidden_channels, n_modes)
            for _ in range(n_layers)
        ])

        # LSTM: processes (batch, n_bins, hidden_channels) sequentially
        # n_bins acts as the sequence length; hidden state encodes momentum
        self.lstm = nn.LSTM(
            input_size=hidden_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True
        )

        # Projection: lstm_hidden → 1 density value per bin
        self.project = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden // 2),
            nn.GELU(),
            nn.Linear(lstm_hidden // 2, 1)
        )

    def forward(
        self,
        rho_window: torch.Tensor,
        lstm_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Predict density at next timestep.

        Args:
            rho_window : Stacked density history, shape (batch, k_history, n_bins)
            lstm_state : Optional (h_n, c_n) from the previous step;
                         if None the LSTM is initialised with zeros.

        Returns:
            rho_out    : Predicted density,      shape (batch, n_bins)
            lstm_state : Updated LSTM state  (h_n, c_n) for the next step
        """
        # rho_window: (batch, k, n_bins) — already in channel-first format
        x = self.lift(rho_window)    # (batch, hidden, n_bins)

        # FNO trunk
        for layer in self.fourier_layers:
            x = layer(x)             # (batch, hidden, n_bins)

        # LSTM over spatial positions: treat n_bins as sequence length
        x = x.permute(0, 2, 1)      # (batch, n_bins, hidden)
        x, lstm_state = self.lstm(x, lstm_state)  # (batch, n_bins, lstm_hidden)

        # Project to density
        x = self.project(x)          # (batch, n_bins, 1)
        rho_out = x.squeeze(-1)      # (batch, n_bins)
        rho_out = F.relu(rho_out)    # non-negative density

        return rho_out, lstm_state

    def predict_trajectory(
        self,
        initial_window: torch.Tensor,
        n_steps: int,
        lstm_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Autoregressively predict density evolution for n_steps steps.

        This is where we bypass the Lyapunov horizon: the FNO predicts
        the coarse-grained density distribution, which evolves smoothly
        even when individual particle trajectories have diverged.

        The LSTM hidden state is threaded through every step so that
        the network can maintain an implicit velocity estimate.

        Args:
            initial_window : First k frames, shape (k_history, n_bins)
                             or (1, k_history, n_bins).
            n_steps        : Number of prediction steps.
            lstm_state     : Optional initial LSTM state; zeros if None.

        Returns:
            Trajectory of *predicted* densities, shape (n_steps + 1, n_bins).
            The first entry is the last frame of initial_window.
        """
        if initial_window.dim() == 2:
            initial_window = initial_window.unsqueeze(0)   # (1, k, n_bins)

        # Record the most-recent frame of the seed window as t=0
        trajectory = [initial_window[:, -1, :]]  # list of (1, n_bins)
        window = initial_window                   # (1, k, n_bins)

        with torch.no_grad():
            for _ in range(n_steps):
                rho_next, lstm_state = self.forward(window, lstm_state)
                trajectory.append(rho_next)       # (1, n_bins)
                # Slide window: drop oldest frame, append new prediction
                window = torch.cat(
                    [window[:, 1:, :], rho_next.unsqueeze(1)], dim=1
                )  # (1, k, n_bins)

        # Stack: (n_steps + 1, n_bins)
        return torch.cat(trajectory, dim=0)

    def compute_loss(
        self,
        inputs:               torch.Tensor,
        targets:              torch.Tensor,
        loss_type:            str  = 'mse',
        use_mass_conservation: bool = False,
        use_positivity:       bool  = False,
        mass_weight:          float = 0.1,
        positivity_weight:    float = 0.01
    ) -> torch.Tensor:
        """
        Compute training loss with optional physics-based auxiliary terms.

        Args:
            inputs  : (batch, k_history, n_bins) — stacked density windows
            targets : (batch, n_bins)             — next-step density
            loss_type             : 'mse' | 'l1'
            use_mass_conservation : add mass-conservation penalty
            use_positivity        : add positivity penalty
            mass_weight           : weight for mass-conservation term
            positivity_weight     : weight for positivity term

        Returns:
            Total loss scalar.

        Reference: methods/paper1_methods.tex lines 307-312
        """
        predictions, _ = self.forward(inputs)  # (batch, n_bins); discard LSTM state

        # Base reconstruction loss
        if loss_type == 'mse':
            reconstruction_loss = F.mse_loss(predictions, targets)
        elif loss_type == 'l1':
            reconstruction_loss = F.l1_loss(predictions, targets)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        total_loss = reconstruction_loss

        # Auxiliary Loss 1: Mass Conservation  L_mass = (∫ρ̂ dx − ∫ρ dx)²
        if use_mass_conservation:
            # Use the most-recent frame of the input window as the reference mass
            input_mass  = inputs[:, -1, :].sum(dim=-1)   # (batch,)
            pred_mass   = predictions.sum(dim=-1)
            target_mass = targets.sum(dim=-1)
            conservation_loss = (
                F.mse_loss(pred_mass, input_mass) +
                F.mse_loss(pred_mass, target_mass)
            )
            total_loss = total_loss + mass_weight * conservation_loss

        # Auxiliary Loss 2: Positivity Constraint  L_pos = ‖min(ρ̂, 0)‖²₂
        if use_positivity:
            negative_values = torch.clamp(predictions, max=0.0)
            positivity_loss = torch.mean(negative_values ** 2)
            total_loss = total_loss + positivity_weight * positivity_loss

        return total_loss



# =============================================================================
# Inverse Problem: Mass Inference from Collision Data
# =============================================================================

class CollisionGraphEncoder(nn.Module):
    """
    Encode collision sequences for the inverse problem.
    
    The challenge: infer particle masses and diameters from
    collision timestamps and momentum transfers alone.
    
    Architecture:
    1. Embed each collision event
    2. Use attention/GNN to capture collision graph structure
    3. Pool to get particle-level representations
    4. Predict mass/diameter for each particle
    
    This is highly novel (as per literature review).
    """
    
    def __init__(
        self,
        n_particles: int,
        embedding_dim: int = 64,
        n_attention_heads: int = 4
    ):
        super(CollisionGraphEncoder, self).__init__()
        
        self.n_particles = n_particles
        self.embedding_dim = embedding_dim
        
        # Event embedding (time, Δp, position) -> embedding_dim
        self.event_encoder = nn.Sequential(
            nn.Linear(3, embedding_dim),
            nn.Tanh()
        )
        
        # Particle embeddings (learned)
        self.particle_embeddings = nn.Parameter(
            torch.randn(n_particles, embedding_dim) / np.sqrt(embedding_dim)
        )
        
        # Attention layer for aggregating collision information
        self.attention = nn.MultiheadAttention(
            embedding_dim,
            n_attention_heads,
            batch_first=True
        )
        
        # Output heads
        self.mass_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, 1),
            nn.Softplus()  # Ensure positive output
        )
        
        self.diameter_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, 1),
            nn.Softplus()  # Ensure positive output
        )
    
    def encode_collisions(
        self,
        times: torch.Tensor,
        momentum_transfers: torch.Tensor,
        positions: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode collision events.
        
        Args:
            times: (n_collisions,)
            momentum_transfers: (n_collisions,)
            positions: (n_collisions,)
            
        Returns:
            Collision embeddings (n_collisions, embedding_dim)
        """
        # Stack features
        features = torch.stack([times, momentum_transfers, positions], dim=-1)
        
        # Encode
        embeddings = self.event_encoder(features)
        
        return embeddings
    
    def forward(
        self,
        times: torch.Tensor,
        momentum_transfers: torch.Tensor,
        positions: torch.Tensor,
        particle_pairs: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Infer particle properties from collision sequence.
        
        Args:
            times: (n_collisions,)
            momentum_transfers: (n_collisions,)
            positions: (n_collisions,)
            particle_pairs: (n_collisions, 2) - which particles collided
            
        Returns:
            Dict with 'masses' and 'diameters' tensors, shape (n_particles,)
        """
        if len(times) == 0:
            # No collisions - return uniform prior
            masses = torch.ones(self.n_particles, device=times.device)
            diameters = torch.ones(self.n_particles, device=times.device)
            return {'masses': masses, 'diameters': diameters}
        
        # Encode all collisions
        collision_embeddings = self.encode_collisions(
            times, momentum_transfers, positions
        )  # (n_collisions, embedding_dim)
        
        # Aggregate embeddings per particle
        particle_updates = torch.zeros(
            self.n_particles, self.embedding_dim,
            device=times.device
        )
        collision_counts = torch.zeros(self.n_particles, device=times.device)
        
        for i in range(len(times)):
            p1, p2 = particle_pairs[i]
            particle_updates[p1] += collision_embeddings[i]
            particle_updates[p2] += collision_embeddings[i]
            collision_counts[p1] += 1
            collision_counts[p2] += 1
        
        # Avoid division by zero
        collision_counts = torch.clamp(collision_counts, min=1.0)
        particle_updates = particle_updates / collision_counts.unsqueeze(-1)
        
        # Combine with learned particle embeddings
        particle_features = self.particle_embeddings + particle_updates
        
        # Add batch dimension for attention
        particle_features = particle_features.unsqueeze(0)  # (1, n_particles, embedding_dim)
        
        # Self-attention to capture particle interactions
        particle_features_attended, _ = self.attention(
            particle_features, particle_features, particle_features
        )
        
        # Remove batch dimension
        particle_features = particle_features_attended.squeeze(0)
        
        # Predict properties
        masses = self.mass_head(particle_features).squeeze(-1)
        diameters = self.diameter_head(particle_features).squeeze(-1)
        
        return {
            'masses': masses,
            'diameters': diameters
        }


# =============================================================================
# Training Utilities
# =============================================================================

def train_fno(
    model: FourierNeuralOperator,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_epochs: int = 100,
    learning_rate: float = 0.001,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    scheduler_patience: int = 10,
    early_stopping_patience: int = 20,
    checkpoint_path: Optional[str] = None,
    use_mass_conservation: bool = False,
    use_positivity: bool = False,
    mass_weight: float = 0.1,
    positivity_weight: float = 0.01
) -> Dict:
    """
    Training loop for the FNO.
    
    Args:
        model: FNO model
        train_loader: Training data loader
        val_loader: Validation data loader
        n_epochs: Number of epochs
        learning_rate: Learning rate
        device: Device to train on
        scheduler_patience: Patience for learning rate scheduler
        early_stopping_patience: Patience for early stopping
        checkpoint_path: Path to save best model
        use_mass_conservation: If True, add mass conservation loss
        use_positivity: If True, add positivity constraint loss
        mass_weight: Weight for mass conservation loss
        positivity_weight: Weight for positivity loss
        
    Returns:
        Training history
    """
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=scheduler_patience
    )
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rates': []
    }
    
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    
    print(f"Training on {device}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    if use_mass_conservation:
        print(f"  Mass conservation: enabled (weight={mass_weight})")
    if use_positivity:
        print(f"  Positivity constraint: enabled (weight={positivity_weight})")
    
    for epoch in range(n_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        n_train_batches = 0
        
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            loss = model.compute_loss(
                inputs, targets,
                use_mass_conservation=use_mass_conservation,
                use_positivity=use_positivity,
                mass_weight=mass_weight,
                positivity_weight=positivity_weight
            )
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            n_train_batches += 1
        
        train_loss /= n_train_batches
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        n_val_batches = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                loss = model.compute_loss(
                    inputs, targets,
                    use_mass_conservation=use_mass_conservation,
                    use_positivity=use_positivity,
                    mass_weight=mass_weight,
                    positivity_weight=positivity_weight
                )
                val_loss += loss.item()
                n_val_batches += 1
        
        val_loss /= n_val_batches
        
        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{n_epochs} - "
                  f"Train Loss: {train_loss:.6f}, "
                  f"Val Loss: {val_loss:.6f}, "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            
            if checkpoint_path:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'history': history
                }, checkpoint_path)
                print(f"  → Saved best model (val_loss: {val_loss:.6f})")
        else:
            epochs_without_improvement += 1
        
        # Early stopping
        if epochs_without_improvement >= early_stopping_patience:
            print(f"\nEarly stopping after {epoch + 1} epochs")
            break
    
    # Load best model
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"\nLoaded best model from epoch {checkpoint['epoch'] + 1}")
    
    return history



def train_collision_encoder(
    model: CollisionGraphEncoder,
    collision_sequences: List[CollisionSequence],
    n_epochs: int = 100,
    learning_rate: float = 0.001,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Dict:
    """
    Training loop for the collision encoder.
    
    Args:
        model: CollisionGraphEncoder model
        collision_sequences: List of collision sequences with ground truth
        n_epochs: Number of epochs
        learning_rate: Learning rate
        device: Device to train on
        
    Returns:
        Training history
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    history = {'loss': []}
    
    print(f"Training CollisionGraphEncoder on {device}")
    
    for epoch in range(n_epochs):
        total_loss = 0.0
        
        for seq in collision_sequences:
            if seq.true_masses is None:
                continue
            
            # Convert to tensors
            times = torch.from_numpy(seq.times).float().to(device)
            momentum_transfers = torch.from_numpy(seq.momentum_transfers).float().to(device)
            positions = torch.from_numpy(seq.positions).float().to(device)
            particle_pairs = torch.from_numpy(seq.particle_pairs).long().to(device)
            true_masses = torch.from_numpy(seq.true_masses).float().to(device)
            true_diameters = torch.from_numpy(seq.true_diameters).float().to(device)
            
            # Forward pass
            optimizer.zero_grad()
            predictions = model(times, momentum_transfers, positions, particle_pairs)
            
            # Compute loss
            mass_loss = F.mse_loss(predictions['masses'], true_masses)
            diameter_loss = F.mse_loss(predictions['diameters'], true_diameters)
            loss = mass_loss + diameter_loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(collision_sequences)
        history['loss'].append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{n_epochs} - Loss: {avg_loss:.6f}")
    
    return history


# =============================================================================
# Evaluation Metrics
# =============================================================================

def evaluate_lyapunov_bypass(
    model: FourierNeuralOperator,
    test_trajectory: DensityTrajectory,
    lyapunov_time: float,
    dataset: DensityDataset,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Dict:
    """
    Evaluate how well the FNO predicts beyond the Lyapunov time.

    The model is seeded with the first k_history frames of the trajectory
    (matching the stacked-input format used during training) and then rolled
    out autoregressively.

    Args:
        model          : Trained FNO model
        test_trajectory: Test trajectory
        lyapunov_time  : Lyapunov time for the system
        dataset        : Dataset — used for denormalization statistics
        device         : Device to run on

    Returns:
        Dict with prediction errors at various time horizons.
    """
    model = model.to(device)
    model.eval()

    k = model.k_history
    n_steps_per_lyapunov = int(lyapunov_time / test_trajectory.delta_t)

    # Seed window: first k frames, shape (k, n_bins)
    seed_frames = test_trajectory.density_fields[:k]                  # (k, n_bins)
    seed_norm   = (seed_frames - dataset.input_mean) / dataset.input_std  # (k, n_bins)
    seed_tensor = torch.from_numpy(seed_norm.astype(np.float32)).to(device)  # (k, n_bins)

    # Predict trajectory (returns (n_steps + 1, n_bins), starting from frame k-1)
    n_predict = test_trajectory.n_timesteps - k
    with torch.no_grad():
        predicted_norm = model.predict_trajectory(seed_tensor, n_steps=n_predict)
        predicted = dataset.denormalize_output(predicted_norm).cpu().numpy()  # (n_predict+1, n_bins)

    # Compute errors at different horizons (measured from t = k-1)
    horizons = [0.5, 1.0, 2.0, 5.0, 10.0]  # Multiples of Lyapunov time
    errors       = {}
    correlations = {}

    for h in horizons:
        n_steps = int(h * n_steps_per_lyapunov)
        true_step = n_steps + (k - 1)  # offset to align with trajectory
        if n_steps < len(predicted) and true_step < test_trajectory.n_timesteps:
            error = np.mean(
                (predicted[n_steps] - test_trajectory.density_fields[true_step]) ** 2
            )
            errors[f'{h}x_lyapunov'] = float(error)

            corr = np.corrcoef(
                predicted[n_steps].flatten(),
                test_trajectory.density_fields[true_step].flatten()
            )[0, 1]
            correlations[f'{h}x_lyapunov'] = float(corr)

    return {
        'mse_by_horizon':         errors,
        'correlation_by_horizon': correlations,
        'lyapunov_time':          lyapunov_time,
        'delta_t':                test_trajectory.delta_t,
        'n_steps_per_lyapunov':   n_steps_per_lyapunov,
        'k_history':              k,
    }


def save_model(
    model: nn.Module,
    path: str,
    metadata: Optional[Dict] = None
):
    """Save model with metadata."""
    save_dict = {
        'model_state_dict': model.state_dict(),
        'model_config': {
            'n_bins': getattr(model, 'n_bins', None),
            'n_modes': getattr(model, 'n_modes', None),
            'hidden_channels': getattr(model, 'hidden_channels', None),
            'n_layers': getattr(model, 'n_layers', None),
        }
    }
    
    if metadata:
        save_dict['metadata'] = metadata
    
    torch.save(save_dict, path)
    print(f"Model saved to {path}")


def load_model(
    path: str,
    model_class: type = FourierNeuralOperator,
    device: str = 'cpu'
) -> Tuple[nn.Module, Dict]:
    """Load model with metadata."""
    checkpoint = torch.load(path, map_location=device)
    
    # Reconstruct model from config
    config = checkpoint['model_config']
    model = model_class(**{k: v for k, v in config.items() if v is not None})
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    metadata = checkpoint.get('metadata', {})
    
    print(f"Model loaded from {path}")
    return model, metadata
