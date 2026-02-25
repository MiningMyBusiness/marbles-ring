"""
2D Fourier Neural Operator for the Tilting Billiards Environment

This module provides:
  - SpectralConv2d      : learnable spectral convolution over a 2D spatial grid
  - FNO2DLayer          : one FNO block (spectral + local + activation)
  - FourierNeuralOperator2D : full FNO model that maps
                              ρ(x, y, t) → ρ(x, y, t + Δt)
  - BilliardsDataset    : PyTorch Dataset wrapping recorded density trajectories
  - train_fno2d         : training loop (mirrors train_fno in fno_pytorch.py)
  - FNODensityPredictor : drop-in replacement for the Monte Carlo DensityPredictor
                          in billiards_controllers.py, backed by a trained
                          FourierNeuralOperator2D

Design notes
------------
The 2D Fourier layers use rfft2 / irfft2, truncating to the first
(n_modes_x, n_modes_y) modes — exactly the 2D generalisation of the
1D construction in fno_pytorch.py.  The action (roll_rate, pitch_rate)
is injected as a 2-channel constant field appended to the density channel
before lifting, giving the model explicit knowledge of the tilt command
used to generate the next density snapshot.

Author: Kiran Bhattacharyya
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class DensityTrajectory2D:
    """
    A trajectory of 2D density fields recorded from the billiards environment.

    Shape conventions
    -----------------
    density_fields : (T, H, W)   — T snapshots on an H×W grid
    actions        : (T-1, 2)    — [roll_rate, pitch_rate] for each step
    time_points    : (T,)        — physical time of each snapshot
    delta_t        : float       — fixed Δt between snapshots
    """
    density_fields: np.ndarray          # (T, H, W)
    time_points: np.ndarray             # (T,)
    delta_t: float
    actions: Optional[np.ndarray] = None  # (T-1, 2); None if uncontrolled
    system_params: Dict = field(default_factory=dict)

    @property
    def n_timesteps(self) -> int:
        return len(self.time_points)

    @property
    def grid_shape(self) -> Tuple[int, int]:
        return self.density_fields.shape[1], self.density_fields.shape[2]

    def get_input_output_pairs(
        self,
        n_steps_ahead: int = 1,
        k_history: int = 4
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Return (inputs, targets, actions) pairs with stacked history windows.

        inputs  : (N, k_history, H, W)  — k consecutive density frames
        targets : (N, H, W)             — density n_steps_ahead after the window
        actions : (N, 2) or None        — action applied at the last window frame
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
        )  # (N, k, H, W)
        targets = self.density_fields[
            k_history - 1 + n_steps_ahead:
            k_history - 1 + n_steps_ahead + n_samples
        ]  # (N, H, W)
        if self.actions is not None:
            # Action executed at the LAST frame of each input window
            acts = self.actions[k_history - 1: k_history - 1 + n_samples]  # (N, 2)
        else:
            acts = None
        return inputs, targets, acts


# =============================================================================
# PYTORCH DATASET
# =============================================================================

class BilliardsDataset(Dataset):
    """
    PyTorch Dataset for 2D billiards density trajectories.

    Each sample:
        input  : (k_history + 2, H, W)
                   — k density channels (history window)
                   + roll_rate channel (constant tile)
                   + pitch_rate channel (constant tile)
        target : (1, H, W) — density at the next timestep

    If no actions are recorded the action channels are zeroed out.
    The stacked k density frames give the network velocity information
    via finite differences that a single snapshot cannot provide.
    """

    def __init__(
        self,
        trajectories: List[DensityTrajectory2D],
        n_steps_ahead: int = 1,
        normalize: bool = True,
        k_history: int = 4
    ):
        self.normalize = normalize
        self.k_history = k_history

        all_inputs:  List[np.ndarray] = []
        all_targets: List[np.ndarray] = []
        all_actions: List[np.ndarray] = []

        for traj in trajectories:
            inputs, targets, acts = traj.get_input_output_pairs(n_steps_ahead, k_history)
            N = len(inputs)

            if acts is None:
                acts = np.zeros((N, 2), dtype=np.float32)

            all_inputs.append(inputs)    # (N, k, H, W)
            all_targets.append(targets)  # (N, H, W)
            all_actions.append(acts)     # (N, 2)

        self.inputs  = np.concatenate(all_inputs,  axis=0).astype(np.float32)  # (N, k, H, W)
        self.targets = np.concatenate(all_targets, axis=0).astype(np.float32)  # (N, H, W)
        self.actions = np.concatenate(all_actions, axis=0).astype(np.float32)  # (N, 2)

        # Scalar normalization over all density values (inputs and targets share the scale)
        if self.normalize:
            self.density_mean = float(self.inputs.mean())
            self.density_std  = float(self.inputs.std()) + 1e-8
        else:
            self.density_mean = 0.0
            self.density_std  = 1.0

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        window_norm = (self.inputs[idx]  - self.density_mean) / self.density_std  # (k, H, W)
        target_norm = (self.targets[idx] - self.density_mean) / self.density_std  # (H, W)

        k, H, W = window_norm.shape
        act = self.actions[idx]  # (2,)

        # Action channels broadcast over the full spatial grid
        roll_channel  = np.full((1, H, W), act[0], dtype=np.float32)
        pitch_channel = np.full((1, H, W), act[1], dtype=np.float32)

        # Stack: k density frames + roll + pitch  → (k+2, H, W)
        input_tensor = np.concatenate(
            [window_norm, roll_channel, pitch_channel], axis=0
        )

        return (
            torch.from_numpy(input_tensor),             # (k+2, H, W)
            torch.from_numpy(target_norm[None, :, :])   # (1, H, W)
        )

    def denormalize(self, normalized: torch.Tensor) -> torch.Tensor:
        """Convert a normalized density tensor back to physical units."""
        return normalized * self.density_std + self.density_mean


# =============================================================================
# SPECTRAL CONVOLUTION (2D)
# =============================================================================

class SpectralConv2d(nn.Module):
    """
    2D spectral convolution layer for FNO.

    Operates in Fourier space:
      1. rfft2 the input
      2. Multiply the first (n_modes_x × n_modes_y) modes by learnable weights
      3. irfft2 back to physical space

    Weights are stored as real tensors of shape
        (in_channels, out_channels, n_modes_x, n_modes_y, 2)
    where the last dimension holds (real, imag) parts.

    Reference: Li et al. (2021) "Fourier Neural Operator for PDEs"
    """

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        n_modes_x:    int,
        n_modes_y:    int,
    ):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.n_modes_x    = n_modes_x
        self.n_modes_y    = n_modes_y

        scale = 1.0 / (in_channels * out_channels)
        # Shape: (in_ch, out_ch, modes_x, modes_y, 2)  — stored real for compat
        self.weights = nn.Parameter(
            torch.randn(in_channels, out_channels, n_modes_x, n_modes_y, 2) * scale
        )

    @staticmethod
    def _cmul(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        Batched complex multiply.

        x : (batch, in_ch, mx, my)  complex
        w : (in_ch, out_ch, mx, my) complex
        -> (batch, out_ch, mx, my)  complex
        """
        return torch.einsum("bimn,iomn->bomn", x, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (batch, in_channels, H, W)
        Returns:
            y : (batch, out_channels, H, W)
        """
        B, C, H, W = x.shape

        # 2D real FFT:  (B, C, H, W//2+1) complex
        x_ft = torch.fft.rfft2(x, norm="ortho")

        # Extract the low-frequency block we care about
        mx, my = self.n_modes_x, self.n_modes_y
        x_ft_low = x_ft[:, :, :mx, :my]   # (B, C, mx, my)

        # Complex weights
        w_c = torch.view_as_complex(self.weights.contiguous())  # (in, out, mx, my)

        # Multiply
        out_ft_low = self._cmul(x_ft_low, w_c)  # (B, out, mx, my)

        # Pad remaining modes with zeros
        out_ft = torch.zeros(B, self.out_channels, H, W // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :mx, :my] = out_ft_low

        # IFFT back
        out = torch.fft.irfft2(out_ft, s=(H, W), norm="ortho")
        return out


# =============================================================================
# FNO 2D LAYER AND FULL MODEL
# =============================================================================

class FNO2DLayer(nn.Module):
    """
    One layer of the 2D Fourier Neural Operator.

    Combines:
      - SpectralConv2d  (global, frequency domain)
      - 1×1 convolution (pointwise, spatial domain)
      - GeLU activation
    """

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        n_modes_x:    int,
        n_modes_y:    int,
    ):
        super().__init__()
        self.spectral = SpectralConv2d(in_channels, out_channels, n_modes_x, n_modes_y)
        self.local    = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.act      = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.spectral(x) + self.local(x))


class FourierNeuralOperator2D(nn.Module):
    """
    Full 2D FNO with stacked history input and LSTM recurrence.

    Architecture
    ------------
    1. **Lift**      : Conv2d(k_history + 2, hidden_channels, kernel=1)
                       — k density channels + roll + pitch action channels
    2. **FNO trunk** : N × FNO2DLayer(hidden, hidden, n_modes_x, n_modes_y)
    3. **LSTM**      : reshape (B, hidden, H, W) → (B, H*W, hidden),
                       LSTM treats each spatial pixel as a sequence step,
                       output (B, H*W, lstm_hidden) → reshape to (B, lstm_hidden, H, W)
    4. **Project**   : Conv2d(lstm_hidden, 1, kernel=1) + ReLU

    Input  : (batch, k_history + 2, H, W)
    Output : (batch, 1, H, W),  plus LSTM state (h_n, c_n)

    The LSTM state is threaded through autoregressive rollouts so that the
    network accumulates an implicit momentum field over time.
    """

    def __init__(
        self,
        grid_h:          int   = 20,
        grid_w:          int   = 20,
        n_modes_x:       int   = 8,
        n_modes_y:       int   = 8,
        hidden_channels: int   = 32,
        n_layers:        int   = 4,
        delta_t:         float = 0.05,
        k_history:       int   = 4,
        lstm_hidden:     int   = 64,
        lstm_layers:     int   = 1,
    ):
        super().__init__()
        self.grid_h          = grid_h
        self.grid_w          = grid_w
        self.n_modes_x       = n_modes_x
        self.n_modes_y       = n_modes_y
        self.hidden_channels = hidden_channels
        self.n_layers        = n_layers
        self.delta_t         = delta_t
        self.k_history       = k_history
        self.lstm_hidden     = lstm_hidden
        self.lstm_layers     = lstm_layers

        # Lifting layer: (k_history + 2) channels → hidden_channels
        self.lift = nn.Conv2d(k_history + 2, hidden_channels, kernel_size=1)

        # FNO blocks
        self.fno_layers = nn.ModuleList([
            FNO2DLayer(hidden_channels, hidden_channels, n_modes_x, n_modes_y)
            for _ in range(n_layers)
        ])

        # LSTM: processes flattened spatial positions as sequence steps
        self.lstm = nn.LSTM(
            input_size=hidden_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True
        )

        # Projection: lstm_hidden → 1 density channel
        self.project = nn.Sequential(
            nn.Conv2d(lstm_hidden, lstm_hidden // 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(lstm_hidden // 2, 1, kernel_size=1),
        )

    def forward(
        self,
        x: torch.Tensor,
        lstm_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x          : (batch, k_history+2, H, W)
                         Channels: [density_t-k+1, …, density_t, roll, pitch]
            lstm_state : Optional (h_n, c_n); zeros if None.
        Returns:
            rho_next   : (batch, 1, H, W) — predicted next density (non-negative)
            lstm_state : Updated (h_n, c_n) for the next call
        """
        B, _, H, W = x.shape

        x = self.lift(x)                   # (B, hidden, H, W)
        for layer in self.fno_layers:
            x = layer(x)                   # (B, hidden, H, W)

        # LSTM over flattened spatial positions
        x_seq = x.permute(0, 2, 3, 1).reshape(B, H * W, self.hidden_channels)  # (B, HW, hidden)
        x_seq, lstm_state = self.lstm(x_seq, lstm_state)                         # (B, HW, lstm_h)
        x = x_seq.reshape(B, H, W, self.lstm_hidden).permute(0, 3, 1, 2)        # (B, lstm_h, H, W)

        x = self.project(x)                # (B, 1, H, W)
        return F.relu(x), lstm_state

    def predict_trajectory(
        self,
        initial_window: torch.Tensor,           # (k, H, W) or (1, k, H, W)
        action_sequence: torch.Tensor,           # (n_steps, 2)
        lstm_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Autoregressively predict density evolution under a fixed action sequence.

        The LSTM state is threaded across steps.  The rolling density buffer
        is updated by dropping the oldest frame and appending the new prediction.

        Returns stacked density fields: (n_steps + 1, 1, H, W).
        The first entry is the last frame of initial_window.
        """
        if initial_window.dim() == 3:
            initial_window = initial_window.unsqueeze(0)   # (1, k, H, W)

        B, k, H, W = initial_window.shape
        trajectory = [initial_window[:, -1:, :, :]]  # list of (1, 1, H, W)
        window = initial_window                        # (1, k, H, W)

        with torch.no_grad():
            for step_idx in range(len(action_sequence)):
                act = action_sequence[step_idx]          # (2,)
                roll_ch  = act[0].expand(B, 1, H, W)
                pitch_ch = act[1].expand(B, 1, H, W)
                x_in = torch.cat([window, roll_ch, pitch_ch], dim=1)  # (B, k+2, H, W)
                rho_next, lstm_state = self.forward(x_in, lstm_state)  # (B, 1, H, W)
                trajectory.append(rho_next)
                # Slide window
                window = torch.cat([window[:, 1:, :, :], rho_next], dim=1)  # (B, k, H, W)

        return torch.cat(trajectory, dim=0)  # (n_steps+1, 1, H, W)

    def compute_loss(
        self,
        inputs:  torch.Tensor,
        targets: torch.Tensor,
        use_mass_conservation: bool = True,
        mass_weight: float = 0.1,
    ) -> torch.Tensor:
        """
        MSE reconstruction loss + optional mass conservation regulariser.

        inputs  : (batch, k_history + 2, H, W)
        targets : (batch, 1, H, W)
        """
        predictions, _ = self.forward(inputs)           # (batch, 1, H, W)
        loss = F.mse_loss(predictions, targets)

        if use_mass_conservation:
            pred_mass  = predictions.sum(dim=(-1, -2))  # (batch, 1)
            # Reference: most recent density channel in the input window
            input_mass = inputs[:, self.k_history - 1: self.k_history, :, :].sum(dim=(-1, -2))
            conservation_loss = F.mse_loss(pred_mass, input_mass)
            loss = loss + mass_weight * conservation_loss

        return loss


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train_fno2d(
    model:                  FourierNeuralOperator2D,
    train_loader:           DataLoader,
    val_loader:             DataLoader,
    n_epochs:               int   = 100,
    learning_rate:          float = 1e-3,
    device:                 str   = 'cuda' if torch.cuda.is_available() else 'cpu',
    scheduler_patience:     int   = 10,
    early_stopping_patience:int   = 20,
    checkpoint_path:        Optional[str] = None,
    use_mass_conservation:  bool  = True,
    mass_weight:            float = 0.1,
) -> Dict:
    """
    Training loop for FourierNeuralOperator2D.

    Mirrors train_fno() in fno_pytorch.py for consistency.

    Returns a dict with keys:
        'train_loss', 'val_loss', 'learning_rates'  (lists over epochs)
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=scheduler_patience, verbose=True
    )

    history: Dict[str, list] = {
        'train_loss': [],
        'val_loss': [],
        'learning_rates': [],
    }

    best_val_loss = float('inf')
    epochs_without_improvement = 0

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Training FNO2D on {device}  |  parameters: {n_params:,}")
    if use_mass_conservation:
        print(f"  Mass conservation regulariser: enabled (weight={mass_weight})")

    for epoch in range(n_epochs):
        # ---- Train ----
        model.train()
        train_loss, n_batches = 0.0, 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = model.compute_loss(
                inputs, targets,
                use_mass_conservation=use_mass_conservation,
                mass_weight=mass_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            n_batches  += 1
        train_loss /= n_batches

        # ---- Validate ----
        model.eval()
        val_loss, n_vbatches = 0.0, 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                loss = model.compute_loss(
                    inputs, targets,
                    use_mass_conservation=use_mass_conservation,
                    mass_weight=mass_weight,
                )
                val_loss  += loss.item()
                n_vbatches += 1
        val_loss /= n_vbatches

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])

        scheduler.step(val_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:4d}/{n_epochs} — "
                  f"train: {train_loss:.6f}  val: {val_loss:.6f}  "
                  f"lr: {optimizer.param_groups[0]['lr']:.2e}")

        # Early stopping + checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            if checkpoint_path is not None:
                torch.save({'model_state': model.state_dict(),
                            'epoch': epoch,
                            'val_loss': val_loss}, checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1} "
                      f"(no improvement for {early_stopping_patience} epochs).")
                break

    # Reload best weights if checkpoint was saved
    if checkpoint_path is not None:
        try:
            ckpt = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(ckpt['model_state'])
            print(f"Loaded best checkpoint (epoch {ckpt['epoch']+1}, "
                  f"val_loss={ckpt['val_loss']:.6f})")
        except FileNotFoundError:
            pass

    return history


# =============================================================================
# CHECKPOINT I/O
# =============================================================================

def save_fno2d(model: FourierNeuralOperator2D, path: str) -> None:
    """Save model weights and hyper-parameters to a .pt file."""
    torch.save({
        'model_state': model.state_dict(),
        'config': {
            'grid_h':          model.grid_h,
            'grid_w':          model.grid_w,
            'n_modes_x':       model.n_modes_x,
            'n_modes_y':       model.n_modes_y,
            'hidden_channels': model.hidden_channels,
            'n_layers':        model.n_layers,
            'delta_t':         model.delta_t,
            'k_history':       model.k_history,
            'lstm_hidden':     model.lstm_hidden,
            'lstm_layers':     model.lstm_layers,
        }
    }, path)


def load_fno2d(path: str, device: str = 'cpu') -> FourierNeuralOperator2D:
    """Load a FourierNeuralOperator2D from a saved checkpoint."""
    ckpt  = torch.load(path, map_location=device)
    model = FourierNeuralOperator2D(**ckpt['config'])
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model


# =============================================================================
# FNO-BACKED DENSITY PREDICTOR
# (drop-in replacement for the Monte Carlo version in billiards_controllers.py)
# =============================================================================

class FNODensityPredictor:
    """
    Density predictor backed by a trained FourierNeuralOperator2D.

    This replaces the ``DensityPredictor`` (Monte Carlo proxy) defined in
    ``billiards_controllers.py``.  The interface is intentionally compatible:

        predictor = FNODensityPredictor(config, model, dataset, device)
        densities = predictor.predict_density(
            initial_density, initial_state, action_sequence, dt
        )

    Parameters
    ----------
    config      : TableConfig from billiards_env.py
    model       : trained FourierNeuralOperator2D (call .eval() first)
    dataset     : BilliardsDataset whose normalization stats were used for training
    device      : 'cpu' or 'cuda'
    resolution  : spatial grid resolution (must match model training)
    """

    def __init__(
        self,
        config,             # billiards_env.TableConfig
        model: FourierNeuralOperator2D,
        dataset: BilliardsDataset,
        device: str = 'cpu',
        resolution: int = 20,
    ):
        self.config     = config
        self.model      = model.to(device).eval()
        self.dataset    = dataset
        self.device     = device
        self.resolution = resolution

        # Pre-compute the grid for density estimation calls
        cfg = config
        x_edges = np.linspace(-cfg.half_length, cfg.half_length, resolution + 1)
        y_edges = np.linspace(-cfg.half_width,  cfg.half_width,  resolution + 1)
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2
        self.XX, self.YY = np.meshgrid(x_centers, y_centers)
        self.sigma = cfg.ball_radius * 2
        self._inv2s2 = 1.0 / (2.0 * self.sigma ** 2)

    def state_to_density(self, state) -> np.ndarray:
        """
        Convert a TableState to a (H, W) density field using a Gaussian kernel.
        Vectorised — does not use Python inner loops.
        """
        density = np.zeros((self.resolution, self.resolution), dtype=np.float32)
        for ball in state.active_balls:
            dist_sq = (self.XX - ball.position[0]) ** 2 + (self.YY - ball.position[1]) ** 2
            density += np.exp(-dist_sq * self._inv2s2).astype(np.float32)
        return density

    def predict_density(
        self,
        initial_density: Optional[np.ndarray],   # (H, W) or None → computed from state
        initial_state,                            # TableState
        action_sequence: np.ndarray,              # (n_steps, 2)
        dt: float = 0.05,
    ) -> List[np.ndarray]:
        """
        Predict density fields at each step of action_sequence.

        A rolling buffer of the last k_history frames is maintained so
        the model always receives a stacked history window as input.
        The LSTM hidden state is threaded across steps to accumulate
        an implicit velocity/momentum representation.

        Parameters
        ----------
        initial_density : optional pre-computed (H, W) density;
                          if None it is computed from initial_state.
        initial_state   : TableState (used when initial_density is None)
        action_sequence : (n_steps, 2) array of [roll_rate, pitch_rate]
        dt              : planning timestep (informational; not used by FNO directly)

        Returns
        -------
        List of (H, W) np.ndarray, one per step in action_sequence.
        """
        if initial_density is None:
            initial_density = self.state_to_density(initial_state)

        k = self.model.k_history
        H, W = initial_density.shape

        # Build the initial rolling buffer by repeating the seed frame k times.
        # The model will learn to exploit differences once real predictions accrue.
        rho_norm_0 = (initial_density - self.dataset.density_mean) / self.dataset.density_std
        # buffer shape: (k, H, W)  — oldest frame first
        buffer = np.stack([rho_norm_0] * k, axis=0).astype(np.float32)

        acts_tensor = torch.from_numpy(action_sequence.astype(np.float32)).to(self.device)  # (n_steps, 2)
        lstm_state  = None
        predicted_densities: List[np.ndarray] = []

        with torch.no_grad():
            for step_idx in range(len(action_sequence)):
                act = acts_tensor[step_idx]   # (2,)

                # Build input: (k+2, H, W)
                win_tensor   = torch.from_numpy(buffer).to(self.device)              # (k, H, W)
                roll_ch  = act[0].expand(1, H, W)
                pitch_ch = act[1].expand(1, H, W)
                x_in = torch.cat(
                    [win_tensor, roll_ch, pitch_ch], dim=0
                ).unsqueeze(0)  # (1, k+2, H, W)

                rho_next, lstm_state = self.model(x_in, lstm_state)  # (1, 1, H, W)

                # Denormalise and convert to numpy
                rho_norm_next = rho_next.squeeze().cpu().numpy()   # (H, W)
                rho_physical  = rho_norm_next * self.dataset.density_std + self.dataset.density_mean
                rho_physical  = np.maximum(rho_physical, 0.0)      # ensure non-negative
                predicted_densities.append(rho_physical)

                # Slide the rolling buffer
                buffer = np.concatenate([buffer[1:], rho_norm_next[None]], axis=0)  # (k, H, W)

        return predicted_densities


# =============================================================================
# BILLIARDS DATASET BUILDER (utility)
# =============================================================================

def collect_billiards_trajectories(
    n_trajectories: int = 200,
    n_steps:        int = 100,
    dt_env:         float = 0.05,
    resolution:     int = 20,
    n_balls:        int = 7,
    difficulty:     str = 'medium',
    controller           = None,   # callable or None → zero action
    seed:           int = 0,
) -> List[DensityTrajectory2D]:
    """
    Utility: collect density-field trajectories from the billiards environment.

    If `controller` is None, a zero-action (flat table) policy is used,
    which is fine for learning the *unforced* density dynamics; pass a
    real controller (e.g. GreedyController) for action-conditioned data.

    Returns a list of DensityTrajectory2D objects suitable for BilliardsDataset.
    """
    # Import here to avoid a circular import at module level
    from billiards_env import TableConfig, create_random_scenario

    config = TableConfig()
    trajectories: List[DensityTrajectory2D] = []

    for traj_idx in range(n_trajectories):
        env = create_random_scenario(n_balls=n_balls, seed=seed + traj_idx,
                                     difficulty=difficulty)
        density_fields: List[np.ndarray] = []
        time_points:    List[float]      = []
        actions_list:   List[np.ndarray] = []

        # Pre-compute grid once
        x_edges = np.linspace(-config.half_length, config.half_length, resolution + 1)
        y_edges = np.linspace(-config.half_width,  config.half_width,  resolution + 1)
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2
        XX, YY = np.meshgrid(x_centers, y_centers)
        sigma = config.ball_radius * 2
        inv2s2 = 1.0 / (2.0 * sigma ** 2)

        def _density(state):
            d = np.zeros((resolution, resolution), dtype=np.float32)
            for ball in state.active_balls:
                dsq = (XX - ball.position[0]) ** 2 + (YY - ball.position[1]) ** 2
                d  += np.exp(-dsq * inv2s2).astype(np.float32)
            return d

        state = env.state
        density_fields.append(_density(state))
        time_points.append(state.time)

        done = False
        for step in range(n_steps):
            if done:
                break
            obs = env.get_observation()
            if controller is not None:
                action = np.asarray(controller(obs), dtype=np.float32)
            else:
                action = np.zeros(2, dtype=np.float32)

            state, _reward, done, _info = env.step(action, n_substeps=10)
            actions_list.append(action)
            density_fields.append(_density(state))
            time_points.append(state.time)

        trajectories.append(DensityTrajectory2D(
            density_fields=np.stack(density_fields, axis=0),  # (T, H, W)
            time_points=np.array(time_points),
            delta_t=dt_env,
            actions=np.stack(actions_list, axis=0) if actions_list else None,
            system_params={'n_balls': n_balls, 'difficulty': difficulty},
        ))

        if (traj_idx + 1) % 20 == 0:
            print(f"  Collected {traj_idx + 1}/{n_trajectories} trajectories")

    return trajectories
