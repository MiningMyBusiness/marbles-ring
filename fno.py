"""
Neural Operators for Impulsive Dynamical Systems

This module implements the machine learning methodology for Paper 1:
"Neural Operators for Density Evolution in Impulsive Dynamical Systems"

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

import numpy as np
from typing import List, Tuple, Optional, Dict, Callable
from dataclasses import dataclass, field
import json

# Note: In production, these would be PyTorch/JAX imports
# For now, we implement the architecture logic in numpy
# to show the structure; actual training requires GPU frameworks


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class DensityTrajectory:
    """
    A trajectory of density fields over time.
    
    This is the training data format:
    - Input: density at time t
    - Target: density at time t + delta_t
    
    For supervised learning of the PF operator.
    """
    time_points: np.ndarray  # Shape: (T,)
    density_fields: np.ndarray  # Shape: (T, n_bins)
    delta_t: float  # Time step between consecutive snapshots
    
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
        n_steps_ahead: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create (input, target) pairs for training.
        
        Args:
            n_steps_ahead: How many steps ahead to predict
            
        Returns:
            inputs: Shape (n_samples, n_bins)
            targets: Shape (n_samples, n_bins)
        """
        n_samples = self.n_timesteps - n_steps_ahead
        inputs = self.density_fields[:n_samples]
        targets = self.density_fields[n_steps_ahead:]
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
# Fourier Neural Operator Components
# =============================================================================

class SpectralConvolution:
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
        n_bins: int    # Spatial resolution
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_modes = n_modes
        self.n_bins = n_bins
        
        # Learnable weights in Fourier space
        # Complex weights for each mode
        # Shape: (in_channels, out_channels, n_modes)
        scale = 1.0 / (in_channels * out_channels)
        self.weights_real = scale * np.random.randn(
            in_channels, out_channels, n_modes
        )
        self.weights_imag = scale * np.random.randn(
            in_channels, out_channels, n_modes
        )
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Apply spectral convolution.
        
        Args:
            x: Input tensor, shape (batch, in_channels, n_bins)
            
        Returns:
            Output tensor, shape (batch, out_channels, n_bins)
        """
        batch_size = x.shape[0]
        
        # FFT along spatial dimension
        x_ft = np.fft.rfft(x, axis=-1)
        
        # Keep only first n_modes
        x_ft = x_ft[..., :self.n_modes]
        
        # Multiply by complex weights
        # (batch, in_ch, modes) × (in_ch, out_ch, modes) -> (batch, out_ch, modes)
        weights_complex = self.weights_real + 1j * self.weights_imag
        
        out_ft = np.zeros(
            (batch_size, self.out_channels, self.n_modes),
            dtype=complex
        )
        
        for i in range(self.in_channels):
            for o in range(self.out_channels):
                out_ft[:, o, :] += x_ft[:, i, :] * weights_complex[i, o, :]
        
        # Pad with zeros for higher modes
        out_ft_padded = np.zeros(
            (batch_size, self.out_channels, self.n_bins // 2 + 1),
            dtype=complex
        )
        out_ft_padded[..., :self.n_modes] = out_ft
        
        # IFFT back to physical space
        out = np.fft.irfft(out_ft_padded, n=self.n_bins, axis=-1)
        
        return out
    
    def get_parameters(self) -> Dict[str, np.ndarray]:
        """Get learnable parameters for optimization."""
        return {
            'weights_real': self.weights_real,
            'weights_imag': self.weights_imag
        }


class FourierLayer:
    """
    One layer of the Fourier Neural Operator.
    
    Combines:
    1. Spectral convolution (global, in Fourier space)
    2. Local linear transform (pointwise)
    3. Nonlinearity (GeLU or ReLU)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes: int,
        n_bins: int
    ):
        self.spectral_conv = SpectralConvolution(
            in_channels, out_channels, n_modes, n_bins
        )
        
        # Local linear transform (1x1 convolution equivalent)
        self.local_weights = np.random.randn(
            in_channels, out_channels
        ) / np.sqrt(in_channels)
        self.bias = np.zeros(out_channels)
        
        self.n_bins = n_bins
    
    def gelu(self, x: np.ndarray) -> np.ndarray:
        """Gaussian Error Linear Unit activation."""
        return 0.5 * x * (1 + np.tanh(
            np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)
        ))
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass.
        
        Args:
            x: Shape (batch, in_channels, n_bins)
            
        Returns:
            Shape (batch, out_channels, n_bins)
        """
        # Spectral path
        x_spectral = self.spectral_conv.forward(x)
        
        # Local path: (batch, in_ch, bins) @ (in_ch, out_ch) -> (batch, out_ch, bins)
        x_local = np.einsum('bci,io->bco', x, self.local_weights)
        x_local = x_local + self.bias[np.newaxis, :, np.newaxis]
        
        # Combine and apply nonlinearity
        x_out = self.gelu(x_spectral + x_local)
        
        return x_out


class FourierNeuralOperator:
    """
    Full Fourier Neural Operator for density evolution.
    
    Architecture:
    1. Lift input density to higher-dimensional channel space
    2. Apply N Fourier layers
    3. Project back to density space
    
    For Paper 1, this learns the map:
        ρ(x, t) → ρ(x, t + Δt)
    
    effectively approximating the Perron-Frobenius operator.
    """
    
    def __init__(
        self,
        n_bins: int = 100,
        n_modes: int = 16,
        hidden_channels: int = 32,
        n_layers: int = 4,
        delta_t: float = 1.0
    ):
        self.n_bins = n_bins
        self.n_modes = n_modes
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        self.delta_t = delta_t
        
        # Lifting layer: 1 channel (density) -> hidden_channels
        self.lift_weights = np.random.randn(
            1, hidden_channels
        ) / np.sqrt(1)
        self.lift_bias = np.zeros(hidden_channels)
        
        # Fourier layers
        self.fourier_layers = [
            FourierLayer(
                hidden_channels, hidden_channels, n_modes, n_bins
            )
            for _ in range(n_layers)
        ]
        
        # Projection layer: hidden_channels -> 1
        self.project_weights = np.random.randn(
            hidden_channels, 1
        ) / np.sqrt(hidden_channels)
        self.project_bias = np.zeros(1)
    
    def forward(self, rho: np.ndarray) -> np.ndarray:
        """
        Predict density at next timestep.
        
        Args:
            rho: Input density, shape (batch, n_bins) or (n_bins,)
            
        Returns:
            Predicted density, same shape as input
        """
        # Handle single sample
        squeeze_output = False
        if rho.ndim == 1:
            rho = rho[np.newaxis, :]
            squeeze_output = True
        
        batch_size = rho.shape[0]
        
        # Add channel dimension: (batch, n_bins) -> (batch, 1, n_bins)
        x = rho[:, np.newaxis, :]
        
        # Lift to hidden channels
        # (batch, 1, bins) @ (1, hidden) -> (batch, hidden, bins)
        x = np.einsum('bci,ih->bhi', x, self.lift_weights)
        x = x + self.lift_bias[np.newaxis, :, np.newaxis]
        
        # Apply Fourier layers
        for layer in self.fourier_layers:
            x = layer.forward(x)
        
        # Project to output
        # (batch, hidden, bins) @ (hidden, 1) -> (batch, 1, bins)
        x = np.einsum('bhi,ho->boi', x, self.project_weights)
        x = x + self.project_bias[np.newaxis, :, np.newaxis]
        
        # Remove channel dimension
        rho_out = x[:, 0, :]
        
        # Ensure non-negative density
        rho_out = np.maximum(rho_out, 0)
        
        if squeeze_output:
            rho_out = rho_out[0]
        
        return rho_out
    
    def predict_trajectory(
        self,
        initial_density: np.ndarray,
        n_steps: int
    ) -> np.ndarray:
        """
        Predict density evolution for multiple steps.
        
        This is where we bypass the Lyapunov horizon!
        
        Args:
            initial_density: Starting density, shape (n_bins,)
            n_steps: Number of steps to predict
            
        Returns:
            Trajectory, shape (n_steps + 1, n_bins)
        """
        trajectory = [initial_density.copy()]
        rho = initial_density
        
        for _ in range(n_steps):
            rho = self.forward(rho)
            trajectory.append(rho.copy())
        
        return np.array(trajectory)
    
    def compute_loss(
        self,
        inputs: np.ndarray,
        targets: np.ndarray
    ) -> float:
        """
        Compute MSE loss for training.
        
        Args:
            inputs: Input densities, shape (batch, n_bins)
            targets: Target densities, shape (batch, n_bins)
            
        Returns:
            Mean squared error
        """
        predictions = self.forward(inputs)
        mse = np.mean((predictions - targets) ** 2)
        return mse


# =============================================================================
# Inverse Problem: Mass Inference from Collision Data
# =============================================================================

class CollisionGraphEncoder:
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
        self.n_particles = n_particles
        self.embedding_dim = embedding_dim
        self.n_attention_heads = n_attention_heads
        
        # Event embedding (time, Δp, position) -> embedding_dim
        self.event_encoder_weights = np.random.randn(
            3, embedding_dim
        ) / np.sqrt(3)
        
        # Particle embedding (learned)
        self.particle_embeddings = np.random.randn(
            n_particles, embedding_dim
        ) / np.sqrt(embedding_dim)
        
        # Output heads
        self.mass_head = np.random.randn(embedding_dim, 1) / np.sqrt(embedding_dim)
        self.diameter_head = np.random.randn(embedding_dim, 1) / np.sqrt(embedding_dim)
    
    def encode_collision(
        self,
        time: float,
        momentum_transfer: float,
        position: float
    ) -> np.ndarray:
        """Encode a single collision event."""
        features = np.array([time, momentum_transfer, position])
        embedding = np.tanh(features @ self.event_encoder_weights)
        return embedding
    
    def forward(self, collision_sequence: CollisionSequence) -> Dict[str, np.ndarray]:
        """
        Infer particle properties from collision sequence.
        
        Returns:
            Dict with 'masses' and 'diameters' arrays
        """
        # Encode all collisions
        collision_embeddings = []
        for i in range(collision_sequence.n_collisions):
            emb = self.encode_collision(
                collision_sequence.times[i],
                collision_sequence.momentum_transfers[i],
                collision_sequence.positions[i]
            )
            collision_embeddings.append(emb)
        
        if not collision_embeddings:
            # No collisions - return prior
            return {
                'masses': np.ones(self.n_particles),
                'diameters': np.ones(self.n_particles)
            }
        
        collision_embeddings = np.array(collision_embeddings)
        
        # Aggregate embeddings per particle
        particle_updates = np.zeros((self.n_particles, self.embedding_dim))
        collision_counts = np.zeros(self.n_particles)
        
        for i in range(collision_sequence.n_collisions):
            p1, p2 = collision_sequence.particle_pairs[i]
            particle_updates[p1] += collision_embeddings[i]
            particle_updates[p2] += collision_embeddings[i]
            collision_counts[p1] += 1
            collision_counts[p2] += 1
        
        # Avoid division by zero
        collision_counts = np.maximum(collision_counts, 1)
        particle_updates /= collision_counts[:, np.newaxis]
        
        # Combine with learned particle embeddings
        particle_features = self.particle_embeddings + particle_updates
        
        # Predict properties (positive outputs via softplus)
        masses = np.log(1 + np.exp(particle_features @ self.mass_head)).flatten()
        diameters = np.log(1 + np.exp(particle_features @ self.diameter_head)).flatten()
        
        return {
            'masses': masses,
            'diameters': diameters
        }
    
    def compute_loss(
        self,
        collision_sequence: CollisionSequence
    ) -> float:
        """Compute loss against ground truth."""
        if collision_sequence.true_masses is None:
            raise ValueError("No ground truth available")
        
        predictions = self.forward(collision_sequence)
        
        mass_loss = np.mean((predictions['masses'] - collision_sequence.true_masses) ** 2)
        diameter_loss = np.mean((predictions['diameters'] - collision_sequence.true_diameters) ** 2)
        
        return mass_loss + diameter_loss


# =============================================================================
# Training Utilities
# =============================================================================

def generate_training_data(
    n_trajectories: int,
    n_particles_range: Tuple[int, int] = (3, 10),
    track_length: float = 1000.0,
    simulation_time: float = 100.0,
    save_interval: float = 0.1,
    n_bins: int = 100,
    seed: Optional[int] = None
) -> List[DensityTrajectory]:
    """
    Generate training data by running many simulations.
    
    For Paper 1, we need diverse trajectories to train
    the neural operator to generalize.
    """
    from .particle_system import create_random_system
    
    if seed is not None:
        np.random.seed(seed)
    
    trajectories = []
    
    for i in range(n_trajectories):
        n_particles = np.random.randint(n_particles_range[0], n_particles_range[1] + 1)
        
        system = create_random_system(
            n_particles=n_particles,
            track_length=track_length,
            seed=seed + i if seed else None
        )
        
        # Run simulation
        states, collisions = system.evolve(
            duration=simulation_time,
            save_interval=save_interval
        )
        
        # Extract density fields
        time_points = np.array([s.time for s in states])
        density_fields = np.array([
            system.compute_density_field(n_bins=n_bins)
            for _ in states  # Note: need to recompute at each saved state
        ])
        
        # Actually, we need to rerun to get densities at each state
        # This is a simplification - in practice, save densities during simulation
        system_copy = create_random_system(
            n_particles=n_particles,
            track_length=track_length,
            seed=seed + i if seed else None
        )
        
        densities = []
        for state in states:
            # Restore state
            for j, p in enumerate(system_copy.particles):
                p.position = state.particles[j].position
                p.velocity = state.particles[j].velocity
            density = system_copy.compute_density_field(n_bins=n_bins)
            densities.append(density)
        
        trajectory = DensityTrajectory(
            time_points=time_points,
            density_fields=np.array(densities),
            delta_t=save_interval,
            system_params={
                'n_particles': n_particles,
                'track_length': track_length,
                'masses': [p.mass for p in system.particles],
                'diameters': [p.diameter for p in system.particles]
            }
        )
        
        trajectories.append(trajectory)
    
    return trajectories


def train_fno(
    fno: FourierNeuralOperator,
    training_data: List[DensityTrajectory],
    n_epochs: int = 100,
    learning_rate: float = 0.001,
    batch_size: int = 32,
    validation_split: float = 0.1
) -> Dict:
    """
    Training loop for the FNO.
    
    Note: This is a simplified numpy implementation.
    Production code would use PyTorch/JAX with GPU acceleration.
    
    Returns training history.
    """
    # Collect all input-output pairs
    all_inputs = []
    all_targets = []
    
    for traj in training_data:
        inputs, targets = traj.get_input_output_pairs(n_steps_ahead=1)
        all_inputs.append(inputs)
        all_targets.append(targets)
    
    all_inputs = np.concatenate(all_inputs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    n_samples = len(all_inputs)
    n_val = int(n_samples * validation_split)
    
    # Split
    indices = np.random.permutation(n_samples)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]
    
    train_inputs = all_inputs[train_indices]
    train_targets = all_targets[train_indices]
    val_inputs = all_inputs[val_indices]
    val_targets = all_targets[val_indices]
    
    history = {
        'train_loss': [],
        'val_loss': []
    }
    
    print(f"Training FNO on {len(train_inputs)} samples...")
    
    for epoch in range(n_epochs):
        # Shuffle training data
        perm = np.random.permutation(len(train_inputs))
        train_inputs = train_inputs[perm]
        train_targets = train_targets[perm]
        
        # Training (simplified - no actual gradient descent in numpy)
        train_loss = fno.compute_loss(train_inputs, train_targets)
        val_loss = fno.compute_loss(val_inputs, val_targets)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{n_epochs} - "
                  f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
    
    return history


# =============================================================================
# Evaluation Metrics
# =============================================================================

def evaluate_lyapunov_bypass(
    fno: FourierNeuralOperator,
    test_trajectory: DensityTrajectory,
    lyapunov_time: float
) -> Dict:
    """
    Evaluate how well the FNO predicts beyond the Lyapunov time.
    
    This is the key metric for Paper 1!
    
    Returns:
        Dict with prediction errors at various time horizons
    """
    n_steps_per_lyapunov = int(lyapunov_time / test_trajectory.delta_t)
    
    initial_density = test_trajectory.density_fields[0]
    
    # Predict trajectory
    predicted = fno.predict_trajectory(
        initial_density,
        n_steps=test_trajectory.n_timesteps - 1
    )
    
    # Compute errors at different horizons
    horizons = [0.5, 1.0, 2.0, 5.0, 10.0]  # Multiples of Lyapunov time
    errors = {}
    
    for h in horizons:
        n_steps = int(h * n_steps_per_lyapunov)
        if n_steps < test_trajectory.n_timesteps:
            error = np.mean((predicted[n_steps] - test_trajectory.density_fields[n_steps]) ** 2)
            errors[f'{h}x_lyapunov'] = error
    
    # Also compute correlation (density profile shape)
    correlations = {}
    for h in horizons:
        n_steps = int(h * n_steps_per_lyapunov)
        if n_steps < test_trajectory.n_timesteps:
            corr = np.corrcoef(
                predicted[n_steps].flatten(),
                test_trajectory.density_fields[n_steps].flatten()
            )[0, 1]
            correlations[f'{h}x_lyapunov'] = corr
    
    return {
        'mse_by_horizon': errors,
        'correlation_by_horizon': correlations,
        'lyapunov_time': lyapunov_time,
        'delta_t': test_trajectory.delta_t
    }
