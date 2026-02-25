"""
Baseline Methods for Paper 1 Experiments

This module implements baseline methods for comparison against the FNO:
1. LSTM/RNN - Sequential density prediction
2. U-Net CNN - Convolutional encoder-decoder
3. EDMD - Extended Dynamic Mode Decomposition (Koopman operator)
4. Ulam's Method - Discretized Perron-Frobenius operator

Reference: methods/paper1_methods.tex lines 396-431
Author: Kiran Bhattacharyya
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from scipy.linalg import pinv
from typing import List, Tuple, Optional
from dataclasses import dataclass


# =============================================================================
# LSTM Baseline (Methods lines 413-421)
# =============================================================================

class LSTMDensityPredictor(nn.Module):
    """
    LSTM baseline for density field prediction.
    
    Architecture:
    - Input: Flattened density field
    - LSTM: 2 layers, hidden dimension 256
    - Output: Next density field
    
    This baseline treats density evolution as a sequence prediction problem.
    """
    
    def __init__(self, n_bins: int = 128, hidden_dim: int = 256, n_layers: int = 2):
        super(LSTMDensityPredictor, self).__init__()
        
        self.n_bins = n_bins
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.lstm = nn.LSTM(
            input_size=n_bins,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=0.1 if n_layers > 1 else 0.0
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_bins)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input density, shape (batch, n_bins)
            
        Returns:
            Predicted density, shape (batch, n_bins)
        """
        # Add sequence dimension: (batch, n_bins) -> (batch, 1, n_bins)
        x = x.unsqueeze(1)
        
        # LSTM forward
        out, _ = self.lstm(x)
        
        # Take last timestep output
        out = out[:, -1, :]  # (batch, hidden_dim)
        
        # Fully connected layers
        out = self.fc(out)
        
        # Ensure non-negative density
        out = F.relu(out)
        
        return out
    
    def predict_trajectory(self, initial_density: torch.Tensor, n_steps: int) -> torch.Tensor:
        """
        Autoregressively predict trajectory.
        
        Args:
            initial_density: Starting density, shape (n_bins,) or (1, n_bins)
            n_steps: Number of steps to predict
            
        Returns:
            Trajectory, shape (n_steps + 1, n_bins)
        """
        if initial_density.dim() == 1:
            initial_density = initial_density.unsqueeze(0)
        
        trajectory = [initial_density]
        rho = initial_density
        
        with torch.no_grad():
            for _ in range(n_steps):
                rho = self.forward(rho)
                trajectory.append(rho)
        
        return torch.cat(trajectory, dim=0)


# =============================================================================
# U-Net CNN Baseline (Methods lines 423-430)
# =============================================================================

class UNet1D(nn.Module):
    """
    1D U-Net for density field prediction.
    
    Architecture:
    - Encoder: 4 convolutional blocks with downsampling
    - Decoder: 4 transposed convolutional blocks
    - Skip connections between encoder and decoder
    
    This baseline captures spatial structure in the density field.
    """
    
    def __init__(self, n_bins: int = 128, base_channels: int = 32):
        super(UNet1D, self).__init__()
        
        self.n_bins = n_bins
        
        # Encoder (downsampling path)
        self.enc1 = self._conv_block(1, base_channels)
        self.enc2 = self._conv_block(base_channels, base_channels * 2)
        self.enc3 = self._conv_block(base_channels * 2, base_channels * 4)
        self.enc4 = self._conv_block(base_channels * 4, base_channels * 8)
        
        self.pool = nn.MaxPool1d(2)
        
        # Bottleneck
        self.bottleneck = self._conv_block(base_channels * 8, base_channels * 16)
        
        # Decoder (upsampling path)
        self.upconv4 = nn.ConvTranspose1d(base_channels * 16, base_channels * 8, 
                                           kernel_size=2, stride=2)
        self.dec4 = self._conv_block(base_channels * 16, base_channels * 8)  # *16 from skip
        
        self.upconv3 = nn.ConvTranspose1d(base_channels * 8, base_channels * 4, 
                                           kernel_size=2, stride=2)
        self.dec3 = self._conv_block(base_channels * 8, base_channels * 4)
        
        self.upconv2 = nn.ConvTranspose1d(base_channels * 4, base_channels * 2, 
                                           kernel_size=2, stride=2)
        self.dec2 = self._conv_block(base_channels * 4, base_channels * 2)
        
        self.upconv1 = nn.ConvTranspose1d(base_channels * 2, base_channels, 
                                           kernel_size=2, stride=2)
        self.dec1 = self._conv_block(base_channels * 2, base_channels)
        
        # Final output layer
        self.out = nn.Conv1d(base_channels, 1, kernel_size=1)
    
    def _conv_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """Create a convolutional block with two conv layers."""
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input density, shape (batch, n_bins)
            
        Returns:
            Predicted density, shape (batch, n_bins)
        """
        # Add channel dimension: (batch, n_bins) -> (batch, 1, n_bins)
        x = x.unsqueeze(1)
        
        # Encoder
        enc1 = self.enc1(x)
        x = self.pool(enc1)
        
        enc2 = self.enc2(x)
        x = self.pool(enc2)
        
        enc3 = self.enc3(x)
        x = self.pool(enc3)
        
        enc4 = self.enc4(x)
        x = self.pool(enc4)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder with skip connections
        x = self.upconv4(x)
        x = torch.cat([x, enc4], dim=1)
        x = self.dec4(x)
        
        x = self.upconv3(x)
        x = torch.cat([x, enc3], dim=1)
        x = self.dec3(x)
        
        x = self.upconv2(x)
        x = torch.cat([x, enc2], dim=1)
        x = self.dec2(x)
        
        x = self.upconv1(x)
        x = torch.cat([x, enc1], dim=1)
        x = self.dec1(x)
        
        # Output
        x = self.out(x)
        
        # Remove channel dimension and ensure non-negative
        x = x.squeeze(1)
        x = F.relu(x)
        
        return x
    
    def predict_trajectory(self, initial_density: torch.Tensor, n_steps: int) -> torch.Tensor:
        """Autoregressively predict trajectory."""
        if initial_density.dim() == 1:
            initial_density = initial_density.unsqueeze(0)
        
        trajectory = [initial_density]
        rho = initial_density
        
        with torch.no_grad():
            for _ in range(n_steps):
                rho = self.forward(rho)
                trajectory.append(rho)
        
        return torch.cat(trajectory, dim=0)


# =============================================================================
# EDMD Baseline (Methods lines 401-407)
# =============================================================================

class EDMDPredictor:
    """
    Extended Dynamic Mode Decomposition.
    
    Approximates the Koopman operator using a dictionary of observables:
    - Fourier basis functions up to mode k=32
    - Identity (raw values)
    - Regularized least-squares fitting
    
    This is a linear method that works well for linear/weakly-nonlinear dynamics.
    """
    
    def __init__(self, n_modes: int = 32, reg_param: float = 1e-6):
        self.n_modes = n_modes
        self.reg_param = reg_param
        self.K = None  # Koopman operator matrix
        self.n_bins = None
    
    def _dictionary(self, x: np.ndarray) -> np.ndarray:
        """
        Compute dictionary of observables.
        
        Args:
            x: Density fields, shape (n_samples, n_bins)
            
        Returns:
            Features, shape (n_samples, n_features)
        """
        # FFT and keep first n_modes
        x_fft = np.fft.rfft(x, axis=-1)[:, :self.n_modes]
        
        # Extract real and imaginary parts
        features = np.column_stack([
            x,  # Identity (raw density values)
            x_fft.real,
            x_fft.imag
        ])
        
        return features
    
    def fit(self, X_t: np.ndarray, X_t1: np.ndarray):
        """
        Fit Koopman operator from data.
        
        Args:
            X_t: Current states, shape (n_samples, n_bins)
            X_t1: Next states, shape (n_samples, n_bins)
        """
        self.n_bins = X_t.shape[1]
        
        # Compute dictionary features
        Psi_t = self._dictionary(X_t)
        Psi_t1 = self._dictionary(X_t1)
        
        # Solve for Koopman operator: Psi_t1 = K @ Psi_t
        # K = Psi_t1 @ pinv(Psi_t) with regularization
        
        # Add regularization: (Psi_t^T @ Psi_t + λI)^{-1}
        G = Psi_t.T @ Psi_t + self.reg_param * np.eye(Psi_t.shape[1])
        A = Psi_t1.T @ Psi_t
        
        self.K = np.linalg.solve(G.T, A.T).T
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict next state.
        
        Args:
            x: Current density, shape (n_samples, n_bins) or (n_bins,)
            
        Returns:
            Predicted density, same shape as input
        """
        if self.K is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Handle single sample
        single_sample = False
        if x.ndim == 1:
            x = x[np.newaxis, :]
            single_sample = True
        
        # Apply Koopman operator
        psi_x = self._dictionary(x)
        psi_x1 = (self.K @ psi_x.T).T
        
        # Reconstruct density from features (use identity part)
        x_pred = psi_x1[:, :self.n_bins]
        
        # Ensure non-negative
        x_pred = np.maximum(x_pred, 0)
        
        if single_sample:
            x_pred = x_pred[0]
        
        return x_pred
    
    def predict_trajectory(self, initial_density: np.ndarray, n_steps: int) -> np.ndarray:
        """Autoregressively predict trajectory."""
        trajectory = [initial_density.copy()]
        rho = initial_density
        
        for _ in range(n_steps):
            rho = self.predict(rho)
            trajectory.append(rho.copy())
        
        return np.array(trajectory)


# =============================================================================
# Ulam's Method Baseline (Methods lines 409-411)
# =============================================================================

class UlamPredictor:
    """
    Ulam's method for discretizing the Perron-Frobenius operator.
    
    Algorithm:
    - Partition phase space into cells (bins)
    - Compute transition probabilities between bins from data
    - Apply transition matrix to evolve density
    
    This is a classical method for approximating the PF operator.
    """
    
    def __init__(self, n_bins: int = 100):
        self.n_bins = n_bins
        self.transition_matrix = None
    
    def fit(self, density_trajectories: List[np.ndarray]):
        """
        Estimate transition matrix from trajectories.
        
        Args:
            density_trajectories: List of density trajectories,
                                  each shape (n_timesteps, n_bins)
        """
        # Initialize transition count matrix
        counts = np.zeros((self.n_bins, self.n_bins))
        
        for traj in density_trajectories:
            for t in range(len(traj) - 1):
                rho_t = traj[t]
                rho_t1 = traj[t + 1]
                
                # Treat density as probability distribution over bins
                # Transition: rho_t -> rho_t1
                # Estimate: P(bin_j | bin_i) ≈ rho_t1[j] if mass in bin i
                
                # Weight transitions by current density
                for i in range(self.n_bins):
                    if rho_t[i] > 1e-10:
                        for j in range(self.n_bins):
                            # Probability of going from i to j
                            counts[j, i] += rho_t[i] * rho_t1[j]
        
        # Normalize columns to get transition probabilities
        col_sums = counts.sum(axis=0, keepdims=True)
        col_sums = np.where(col_sums < 1e-10, 1.0, col_sums)  # Avoid division by zero
        
        self.transition_matrix = counts / col_sums
        
        # Add small amount of noise for stability (Laplace smoothing)
        self.transition_matrix = (self.transition_matrix + 1e-8) / (1 + 1e-8 * self.n_bins)
    
    def predict(self, rho: np.ndarray) -> np.ndarray:
        """
        Apply transition matrix to evolve density.
        
        Args:
            rho: Current density, shape (n_bins,) or (n_samples, n_bins)
            
        Returns:
            Next density, same shape as input
        """
        if self.transition_matrix is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Handle batch
        if rho.ndim == 1:
            rho_next = self.transition_matrix @ rho
        else:
            rho_next = (self.transition_matrix @ rho.T).T
        
        # Ensure non-negative and normalized
        rho_next = np.maximum(rho_next, 0)
        
        return rho_next
    
    def predict_trajectory(self, initial_density: np.ndarray, n_steps: int) -> np.ndarray:
        """Autoregressively predict trajectory."""
        trajectory = [initial_density.copy()]
        rho = initial_density
        
        for _ in range(n_steps):
            rho = self.predict(rho)
            trajectory.append(rho.copy())
        
        return np.array(trajectory)


# =============================================================================
# Training Utilities
# =============================================================================

def train_baseline_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_epochs: int = 50,
    learning_rate: float = 1e-3,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> dict:
    """
    Train a PyTorch baseline model (LSTM or U-Net).
    
    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        n_epochs: Number of training epochs
        learning_rate: Learning rate
        device: Device to train on
        
    Returns:
        Training history dictionary
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    
    for epoch in range(n_epochs):
        # Training
        model.train()
        train_loss = 0.0
        n_train = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            n_train += inputs.size(0)
        
        train_loss /= n_train
        
        # Validation
        model.eval()
        val_loss = 0.0
        n_val = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item() * inputs.size(0)
                n_val += inputs.size(0)
        
        val_loss /= n_val
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1}/{n_epochs}: "
                  f"Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
    
    return history
