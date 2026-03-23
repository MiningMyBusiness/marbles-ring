"""
Experiment Scripts for Paper 1:
"Neural Operators for Density Evolution in Impulsive Dynamical Systems"

This module contains all experiments needed for the ML methodology paper.

Key experiments:
1. Dataset generation (diverse 1D hard-sphere systems)
2. Lyapunov exponent characterization
3. FNO training and validation
4. Lyapunov horizon bypass demonstration
5. Ablation studies
6. Baseline comparisons
7. Inverse problem experiments

Each experiment is designed to produce a specific figure or table.

Author: Kiran Bhattacharyya
"""

import numpy as np
import json
import os
import time
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import sys
from multiprocessing import Pool, cpu_count
from functools import partial


# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from particle_system import (
    ParticleSystem, create_random_system, validate_conservation,
    SystemState, CollisionEvent
)

# Use PyTorch implementation
from fno_pytorch import (
    FourierNeuralOperator, DensityTrajectory, CollisionSequence,
    CollisionGraphEncoder, evaluate_lyapunov_bypass,
    DensityDataset, train_fno, train_collision_encoder,
    save_model, load_model
)

# Baseline methods
from baselines import (
    LSTMDensityPredictor, UNet1D, EDMDPredictor, UlamPredictor,
    train_baseline_model
)



# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================

@dataclass
class ExperimentConfig:
    """
    Configuration for reproducible experiments.
    
    Default values match the specifications in methods/paper1_methods.tex:
    - Table 1 (Architecture): n=128, k_max=32, d_h=64, T=4
    - Table 2 (Dataset): N∈[3,15], d∈[5,25], |v|∈[20,80], sim_time=500
    """
    experiment_name: str
    seed: int = 42
    output_dir: str = "./results/paper1"
    
    # System parameters (Table 2, lines 274-292)
    track_length: float = 1000.0  # Fixed
    n_particles_range: Tuple[int, int] = (3, 15)  # Was (3, 10)
    diameter_range: Tuple[float, float] = (5.0, 25.0)  # Was (5.0, 20.0)
    speed_range: Tuple[float, float] = (20.0, 80.0)  # Was (10.0, 50.0)
    
    # Simulation parameters (Table 2)
    simulation_time: float = 500.0  # Was 100.0
    save_interval: float = 0.1  # Δt in paper
    n_density_bins: int = 128  # Was 100 (Table 1: spatial resolution n=128)
    
    # Training parameters (Section 4.2, line 296)
    n_training_trajectories: int = 1000  # Was 500 (yields ~5M samples)
    n_validation_trajectories: int = 100  # Scaled up proportionally
    n_test_trajectories: int = 100  # Scaled up proportionally
    
    # FNO architecture (Table 1, lines 206-221)
    fno_modes: int = 32  # Was 16 (k_max in paper)
    fno_hidden_channels: int = 64  # Was 32 (d_h in paper)
    fno_layers: int = 4  # Matches paper (T in paper)

    # Stacked history window for FNO input (resolves ill-posedness from 
    # missing momentum information in a single density snapshot)
    k_history: int = 4   # Number of past frames stacked as FNO input

    # LSTM recurrence
    lstm_hidden: int = 64   # Hidden dim of the LSTM appended after FNO trunk
    lstm_layers: int = 1    # Number of stacked LSTM layers

    # Auxiliary loss terms (Methods lines 307-312)
    use_mass_conservation: bool = False  # Enable mass conservation loss
    use_positivity: bool = False  # Enable positivity constraint loss
    mass_weight: float = 0.1  # Weight for mass conservation loss
    positivity_weight: float = 0.01  # Weight for positivity loss
    
    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'ExperimentConfig':
        with open(path, 'r') as f:
            return cls(**json.load(f))



# =============================================================================
# EXPERIMENT 1: DATASET GENERATION
# =============================================================================

def _generate_single_trajectory(args: Tuple) -> Dict:
    """
    Generate a single trajectory (worker function for multiprocessing).
    
    Args:
        args: Tuple of (i, split_name, config)
        
    Returns:
        Trajectory data dictionary
    """
    i, split_name, config = args
    
    # Random number of particles
    n_particles = np.random.randint(
        config.n_particles_range[0],
        config.n_particles_range[1] + 1
    )
    
    # Create system
    system = create_random_system(
        n_particles=n_particles,
        track_length=config.track_length,
        diameter_range=config.diameter_range,
        speed_range=config.speed_range,
        seed=config.seed + i + (hash(split_name) % 10000)
    )
    
    # Record initial state
    initial_masses = [p.mass for p in system.particles]
    initial_diameters = [p.diameter for p in system.particles]
    initial_velocities = [p.velocity for p in system.particles]
    initial_positions = [p.position for p in system.particles]
    
    # Run simulation
    states, collisions = system.evolve(
        duration=config.simulation_time,
        save_interval=config.save_interval
    )
    
    # Compute density fields at each saved state
    density_fields = []
    for state in states:
        # Restore state to compute density
        for j, p in enumerate(system.particles):
            p.position = state.particles[j].position
            p.velocity = state.particles[j].velocity
        density = system.compute_density_field(n_bins=config.n_density_bins)
        density_fields.append(density.tolist())
    
    # Validate conservation
    conservation = validate_conservation(states)
    
    trajectory_data = {
        'id': f"{split_name}_{i}",
        'n_particles': n_particles,
        'masses': initial_masses,
        'diameters': initial_diameters,
        'initial_positions': initial_positions,
        'initial_velocities': initial_velocities,
        'time_points': [s.time for s in states],
        'density_fields': density_fields,
        'n_collisions': len(collisions),
        'collision_times': [c.time for c in collisions],
        'energy_conserved': conservation['energy_conserved'],
        'momentum_conserved': conservation['momentum_conserved'],
    }
    
    return trajectory_data


def experiment_1_generate_dataset(config: ExperimentConfig, n_workers: Optional[int] = None) -> Dict:
    """
    Generate training, validation, and test datasets (parallelized).
    
    Figure 1 (supplementary): Dataset statistics
    - Distribution of particle counts
    - Distribution of masses/diameters
    - Distribution of collision rates
    
    Output: Saved trajectories for training
    
    Args:
        config: Experiment configuration
        n_workers: Number of parallel workers (default: CPU count - 1)
    """
    print("=" * 60)
    print("EXPERIMENT 1: Dataset Generation")
    print("=" * 60)
    
    np.random.seed(config.seed)
    
    # Determine number of workers
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)  # Leave one CPU free
    
    print(f"\nUsing {n_workers} parallel workers (CPU count: {cpu_count()})")
    
    results = {
        'train': [],
        'val': [],
        'test': [],
        'statistics': {}
    }
    
    def generate_trajectories_parallel(n_trajectories: int, split_name: str) -> List[Dict]:
        """Generate trajectories in parallel using multiprocessing."""
        # Prepare arguments for each trajectory
        args_list = [(i, split_name, config) for i in range(n_trajectories)]
        
        # Use multiprocessing Pool
        trajectories = []
        with Pool(processes=n_workers) as pool:
            # Use imap for progress tracking
            for idx, traj in enumerate(pool.imap(_generate_single_trajectory, args_list)):
                trajectories.append(traj)
                if (idx + 1) % 50 == 0 or idx == 0:
                    print(f"  {split_name}: Generated {idx + 1}/{n_trajectories} trajectories")
        
        return trajectories
    
    # Generate datasets
    print("\nGenerating training set...")
    results['train'] = generate_trajectories_parallel(config.n_training_trajectories, 'train')
    
    print("\nGenerating validation set...")
    results['val'] = generate_trajectories_parallel(config.n_validation_trajectories, 'val')
    
    print("\nGenerating test set...")
    results['test'] = generate_trajectories_parallel(config.n_test_trajectories, 'test')
    
    # Compute statistics
    all_trajectories = results['train'] + results['val'] + results['test']
    
    results['statistics'] = {
        'total_trajectories': len(all_trajectories),
        'total_collisions': sum(t['n_collisions'] for t in all_trajectories),
        'avg_particles': np.mean([t['n_particles'] for t in all_trajectories]),
        'avg_collisions_per_trajectory': np.mean([t['n_collisions'] for t in all_trajectories]),
        'collision_rate_mean': np.mean([
            t['n_collisions'] / config.simulation_time for t in all_trajectories
        ]),
        'particle_count_distribution': {
            str(k): sum(1 for t in all_trajectories if t['n_particles'] == k)
            for k in range(config.n_particles_range[0], config.n_particles_range[1] + 1)
        },
        'conservation_check': {
            'energy_conserved': all(t['energy_conserved'] for t in all_trajectories),
            'momentum_conserved': all(t['momentum_conserved'] for t in all_trajectories)
        }
    }
    
    print("\n" + "-" * 40)
    print("Dataset Statistics:")
    print(f"  Total trajectories: {results['statistics']['total_trajectories']}")
    print(f"  Total collisions: {results['statistics']['total_collisions']}")
    print(f"  Avg particles per system: {results['statistics']['avg_particles']:.2f}")
    print(f"  Avg collisions per trajectory: {results['statistics']['avg_collisions_per_trajectory']:.1f}")
    print(f"  Collision rate: {results['statistics']['collision_rate_mean']:.2f} collisions/time unit")
    
    print("\nParticle count distribution:")
    for n, count in sorted(results['statistics']['particle_count_distribution'].items()):
        print(f"  {n} particles: {count} trajectories")
    
    print("\nConservation laws:")
    print(f"  Energy conserved: {results['statistics']['conservation_check']['energy_conserved']}")
    print(f"  Momentum conserved: {results['statistics']['conservation_check']['momentum_conserved']}")
    
    return results



# =============================================================================
# EXPERIMENT 2: LYAPUNOV EXPONENT CHARACTERIZATION
# =============================================================================

def _compute_lyapunov_single(args: Tuple) -> float:
    """Worker function for parallel Lyapunov computation."""
    n, trial, config = args
    system = create_random_system(
        n_particles=n,
        track_length=config.track_length,
        seed=config.seed + trial + n * 100
    )
    return system.compute_lyapunov_exponent(perturbation=1e-8, duration=50.0)


def _compute_lyapunov_hetero(args: Tuple) -> float:
    """Worker function for heterogeneity Lyapunov computation."""
    hetero, trial, config = args
    system = ParticleSystem(config.track_length)
    n_particles = 5
    base_mass = 1.0
    base_diameter = 10.0
    spacing = config.track_length / n_particles
    
    np.random.seed(config.seed + trial + int(hetero * 1000))
    
    for i in range(n_particles):
        if hetero == 0:
            mass = base_mass
            diameter = base_diameter
        else:
            mass = base_mass * (1 + hetero * (np.random.random() - 0.5) * 2)
            diameter = base_diameter * (mass ** (1/3))
        
        system.add_particle(
            position=spacing * i + spacing * 0.5,
            velocity=(20 + np.random.random() * 40) * (1 if np.random.random() > 0.5 else -1),
            diameter=diameter,
            mass=mass
        )
    
    return system.compute_lyapunov_exponent(perturbation=1e-8, duration=50.0)


def experiment_2_lyapunov_characterization(
    config: ExperimentConfig,
    use_parallel: bool = False,
    n_workers: Optional[int] = None
) -> Dict:
    """
    Characterize chaos in the 1D hard-sphere system.
    
    Figure 2: Lyapunov exponent analysis
    - (a) λ vs number of particles
    - (b) λ vs mass heterogeneity
    - (c) Trajectory divergence over time
    - (d) Lyapunov time distribution
    
    Key result: Show that unequal masses create chaos (λ > 0)
    while equal masses don't (λ ≈ 0).
    
    Args:
        config: Experiment configuration
        use_parallel: If True, parallelize independent computations
        n_workers: Number of workers for parallelization
    """
    print("=" * 60)
    print("EXPERIMENT 2: Lyapunov Exponent Characterization")
    print("=" * 60)
    
    if use_parallel:
        if n_workers is None:
            n_workers = max(1, cpu_count() - 1)
        print(f"\nUsing parallel execution with {n_workers} workers")
    
    np.random.seed(config.seed)
    
    results = {
        'lyapunov_vs_n_particles': [],
        'lyapunov_vs_heterogeneity': [],
        'trajectory_divergence': [],
        'equal_vs_unequal_mass': {}
    }
    
    # Part A: Lyapunov vs number of particles
    print("\nPart A: Lyapunov exponent vs particle count...")
    for n in range(2, 11):
        if use_parallel:
            # Parallel computation
            args_list = [(n, trial, config) for trial in range(10)]
            with Pool(processes=n_workers) as pool:
                lyapunov_samples = pool.map(_compute_lyapunov_single, args_list)
        else:
            # Sequential computation
            lyapunov_samples = []
            for trial in range(10):
                system = create_random_system(
                    n_particles=n,
                    track_length=config.track_length,
                    seed=config.seed + trial + n * 100
                )
                lyap = system.compute_lyapunov_exponent(perturbation=1e-8, duration=50.0)
                lyapunov_samples.append(lyap)
        
        results['lyapunov_vs_n_particles'].append({
            'n_particles': n,
            'lyapunov_mean': float(np.mean(lyapunov_samples)),
            'lyapunov_std': float(np.std(lyapunov_samples)),
            'lyapunov_samples': lyapunov_samples
        })
        print(f"  N={n}: λ = {np.mean(lyapunov_samples):.4f} ± {np.std(lyapunov_samples):.4f}")
    
    # Part B: Lyapunov vs mass heterogeneity
    print("\nPart B: Lyapunov exponent vs mass heterogeneity...")
    heterogeneity_levels = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0]
    
    for hetero in heterogeneity_levels:
        if use_parallel:
            # Parallel computation
            args_list = [(hetero, trial, config) for trial in range(10)]
            with Pool(processes=n_workers) as pool:
                lyapunov_samples = pool.map(_compute_lyapunov_hetero, args_list)
        else:
            # Sequential computation
            lyapunov_samples = []
            for trial in range(10):
                system = ParticleSystem(config.track_length)
                n_particles = 5
                base_mass = 1.0
                base_diameter = 10.0
                spacing = config.track_length / n_particles
                
                for i in range(n_particles):
                    if hetero == 0:
                        mass = base_mass
                        diameter = base_diameter
                    else:
                        mass = base_mass * (1 + hetero * (np.random.random() - 0.5) * 2)
                        mass = max(mass, 0.1 * base_mass)  # clamp to positive
                        diameter = base_diameter * (mass ** (1/3))
                    
                    system.add_particle(
                        position=spacing * i + spacing * 0.5,
                        velocity=(20 + np.random.random() * 40) * (1 if np.random.random() > 0.5 else -1),
                        diameter=diameter,
                        mass=mass
                    )
                
                lyap = system.compute_lyapunov_exponent(perturbation=1e-8, duration=50.0)
                lyapunov_samples.append(lyap)
        
        results['lyapunov_vs_heterogeneity'].append({
            'heterogeneity': hetero,
            'lyapunov_mean': float(np.mean(lyapunov_samples)),
            'lyapunov_std': float(np.std(lyapunov_samples))
        })
        print(f"  Heterogeneity={hetero}: λ = {np.mean(lyapunov_samples):.4f} ± {np.std(lyapunov_samples):.4f}")
    
    # Part C: Trajectory divergence over time (not parallelized - single trajectory analysis)
    print("\nPart C: Trajectory divergence dynamics...")
    system = create_random_system(
        n_particles=5,
        track_length=config.track_length,
        seed=config.seed
    )
    
    # Save initial state
    initial_state = system.get_state()
    perturbation = 1e-8
    
    # Run original trajectory
    original_trajectory = []
    system_copy = create_random_system(n_particles=5, track_length=config.track_length, seed=config.seed)
    
    for t_step in range(100):
        states, _ = system_copy.evolve(duration=1.0, save_interval=1.0)
        original_trajectory.append([p.position for p in system_copy.particles])
    
    # Run perturbed trajectory
    system_perturbed = create_random_system(n_particles=5, track_length=config.track_length, seed=config.seed)
    system_perturbed.particles[0].position += perturbation
    
    perturbed_trajectory = []
    divergences = []
    
    for t_step in range(100):
        states, _ = system_perturbed.evolve(duration=1.0, save_interval=1.0)
        perturbed_trajectory.append([p.position for p in system_perturbed.particles])
        
        # Compute divergence
        div = np.sqrt(sum(
            (original_trajectory[t_step][i] - perturbed_trajectory[t_step][i])**2
            for i in range(5)
        ))
        divergences.append(float(div))
    
    results['trajectory_divergence'] = {
        'time_points': list(range(100)),
        'divergences': divergences,
        'initial_perturbation': perturbation

    }
    
    # Part D: Equal vs unequal mass comparison
    print("\nPart D: Equal vs unequal mass comparison...")
    
    # Equal mass system
    equal_mass_lyapunovs = []
    for trial in range(20):
        system = ParticleSystem(config.track_length)
        for i in range(5):
            system.add_particle(
                position=200 * i + 100,
                velocity=(30 + np.random.random() * 20) * (1 if np.random.random() > 0.5 else -1),
                diameter=10.0,
                mass=1.0  # All equal
            )
        lyap = system.compute_lyapunov_exponent(perturbation=1e-8, duration=50.0)
        equal_mass_lyapunovs.append(lyap)
    
    # Unequal mass system
    unequal_mass_lyapunovs = []
    for trial in range(20):
        system = create_random_system(n_particles=5, track_length=config.track_length, seed=config.seed + trial)
        lyap = system.compute_lyapunov_exponent(perturbation=1e-8, duration=50.0)
        unequal_mass_lyapunovs.append(lyap)
    
    results['equal_vs_unequal_mass'] = {
        'equal_mass': {
            'lyapunov_mean': float(np.mean(equal_mass_lyapunovs)),
            'lyapunov_std': float(np.std(equal_mass_lyapunovs)),
            'samples': equal_mass_lyapunovs
        },
        'unequal_mass': {
            'lyapunov_mean': float(np.mean(unequal_mass_lyapunovs)),
            'lyapunov_std': float(np.std(unequal_mass_lyapunovs)),
            'samples': unequal_mass_lyapunovs
        }
    }
    
    print(f"  Equal mass: λ = {np.mean(equal_mass_lyapunovs):.4f} ± {np.std(equal_mass_lyapunovs):.4f}")
    print(f"  Unequal mass: λ = {np.mean(unequal_mass_lyapunovs):.4f} ± {np.std(unequal_mass_lyapunovs):.4f}")
    
    return results


# =============================================================================
# EXPERIMENT 3: FNO TRAINING
# =============================================================================

def experiment_3_fno_training(
    config: ExperimentConfig,
    dataset: Dict
) -> Dict:
    """
    Train the Fourier Neural Operator on density evolution.
    
    Figure 3: Training dynamics
    - (a) Training and validation loss curves
    - (b) Prediction examples at different time horizons
    
    Table 1: Hyperparameter sensitivity
    """
    print("=" * 60)
    print("EXPERIMENT 3: FNO Training")
    print("=" * 60)
    
    results = {
        'training_history': [],
        'final_metrics': {},
        'hyperparameter_study': []
    }
    
    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    # Convert dataset to DensityTrajectory objects
    print("\nPreparing training data...")
    
    def convert_to_trajectories(traj_list: List[Dict]) -> List[DensityTrajectory]:
        trajectories = []
        for traj in traj_list:
            dt = DensityTrajectory(
                time_points=np.array(traj['time_points']),
                density_fields=np.array(traj['density_fields']),
                delta_t=config.save_interval,
                system_params={
                    'n_particles': traj['n_particles'],
                    'masses': traj['masses'],
                    'diameters': traj['diameters']
                }
            )
            trajectories.append(dt)
        return trajectories
    
    train_trajectories = convert_to_trajectories(dataset['train'])
    val_trajectories = convert_to_trajectories(dataset['val'])
    
    # Create PyTorch datasets (stacked k-frame history windows)
    train_dataset = DensityDataset(train_trajectories, n_steps_ahead=1, normalize=True,
                                   k_history=config.k_history)
    val_dataset   = DensityDataset(val_trajectories,   n_steps_ahead=1, normalize=True,
                                   k_history=config.k_history)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=0  # Set to 0 for compatibility, increase for speed
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0
    )
    
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    
    # Main training run
    print("\nTraining FNO with default hyperparameters...")
    
    fno = FourierNeuralOperator(
        n_bins=config.n_density_bins,
        n_modes=config.fno_modes,
        hidden_channels=config.fno_hidden_channels,
        n_layers=config.fno_layers,
        delta_t=config.save_interval,
        k_history=config.k_history,
        lstm_hidden=config.lstm_hidden,
        lstm_layers=config.lstm_layers
    )
    
    # Train using PyTorch
    checkpoint_path = os.path.join(config.output_dir, 'fno_checkpoint.pt')
    
    history = train_fno(
        model=fno,
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=100,
        learning_rate=1e-3,
        device=device,
        scheduler_patience=10,
        early_stopping_patience=20,
        checkpoint_path=checkpoint_path,
        use_mass_conservation=config.use_mass_conservation,
        use_positivity=config.use_positivity,
        mass_weight=config.mass_weight,
        positivity_weight=config.positivity_weight
    )

    
    # Convert history to serializable format
    results['training_history'] = [
        {
            'epoch': i,
            'train_loss': float(history['train_loss'][i]),
            'val_loss': float(history['val_loss'][i]),
            'learning_rate': float(history['learning_rates'][i])
        }
        for i in range(len(history['train_loss']))
    ]
    
    results['final_metrics'] = {
        'final_train_loss': float(history['train_loss'][-1]),
        'final_val_loss': float(history['val_loss'][-1]),
        'best_val_loss': float(min(history['val_loss'])),
        'n_parameters': sum(p.numel() for p in fno.parameters()),
        'device_used': device
    }
    
    print(f"\n  Final train loss: {results['final_metrics']['final_train_loss']:.6f}")
    print(f"  Final val loss: {results['final_metrics']['final_val_loss']:.6f}")
    print(f"  Best val loss: {results['final_metrics']['best_val_loss']:.6f}")
    print(f"  Model parameters: {results['final_metrics']['n_parameters']:,}")
    
    # Hyperparameter study
    print("\nHyperparameter sensitivity study...")
    
    hyperparams_to_test = [
        {'n_modes': 8, 'hidden_channels': 16, 'n_layers': 2},
        {'n_modes': 16, 'hidden_channels': 32, 'n_layers': 4},  # Default
        {'n_modes': 32, 'hidden_channels': 64, 'n_layers': 4},
        {'n_modes': 16, 'hidden_channels': 32, 'n_layers': 6},
    ]
    
    for hp in hyperparams_to_test:
        print(f"  Testing: {hp}")
        fno_test = FourierNeuralOperator(
            n_bins=config.n_density_bins,
            n_modes=hp['n_modes'],
            hidden_channels=hp['hidden_channels'],
            n_layers=hp['n_layers'],
            k_history=config.k_history,
            lstm_hidden=config.lstm_hidden,
            lstm_layers=config.lstm_layers
        ).to(device)
        
        # Evaluate initial loss
        fno_test.eval()
        val_losses = []
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                loss = fno_test.compute_loss(inputs, targets)
                val_losses.append(loss.item())
                if len(val_losses) >= 10:  # Sample first 10 batches
                    break
        
        avg_loss = np.mean(val_losses)
        
        results['hyperparameter_study'].append({
            **hp,
            'initial_val_loss': float(avg_loss),
            'n_parameters': sum(p.numel() for p in fno_test.parameters())
        })
        print(f"    Initial val loss = {avg_loss:.6f}, Params = {results['hyperparameter_study'][-1]['n_parameters']:,}")
    
    # Return the trained model and dataset for later use
    return results, fno, train_dataset


# =============================================================================
# EXPERIMENT 3b: CURRICULUM FINE-TUNING (SCHEDULED SAMPLING)
# =============================================================================

def experiment_3b_fno_curriculum_finetuning(
    config: ExperimentConfig,
    fno: FourierNeuralOperator,
    dataset: Dict,
    train_dataset: DensityDataset,
) -> Tuple[Dict, FourierNeuralOperator]:
    """
    Fine-tune the pre-trained FNO using scheduled sampling to mitigate the
    covariate shift that accumulates during autoregressive rollout.

    Problem
    -------
    Experiment 3 trains on (ground-truth_t → ground-truth_t+1) pairs.
    At test time (Experiment 4) the model is rolled out autoregressively,
    feeding its own outputs back as inputs.  Small one-step errors compound
    quickly because the model has never seen its own outputs at training time.

    Solution: Scheduled Sampling (Bengio et al., 2015 / DAgger-style)
    -----------------------------------------------------------------
    During fine-tuning each input *frame* in the k-history window is drawn
    from either the ground-truth trajectory or the model's own rollout
    with probability p_gt (teacher-forcing rate).  p_gt is annealed
    linearly from 1.0 (pure teacher forcing, identical to Exp 3) down to
    0.5 (equal mix) over N_ROUNDS curriculum rounds, then held at 0.5.

    Validation and test metrics are reported at p_gt = 0.0 (FNO-only input)
    to directly measure roll-out robustness.

    Figure 3b: Curriculum fine-tuning dynamics
    - (a) Fine-tuning loss curves per curriculum round
    - (b) Val MSE (FNO-input) vs curriculum round
    - (c) Comparison: original FNO vs curriculum-FNO val MSE (FNO-input)

    Args
    ----
    config       : ExperimentConfig
    fno          : Trained FourierNeuralOperator from experiment_3_fno_training
    dataset      : Full dataset dict (train / val / test splits)
    train_dataset: Fitted DensityDataset (provides normalisation stats)

    Returns
    -------
    (results, fno_finetuned)
    """
    import copy
    from torch.utils.data import Dataset as TorchDataset

    print("=" * 60)
    print("EXPERIMENT 3b: Curriculum Fine-Tuning (Scheduled Sampling)")
    print("=" * 60)

    # ── Hyperparameters ────────────────────────────────────────────────────
    N_ROUNDS           = 5        # Curriculum rounds
    EPOCHS_PER_ROUND   = 20       # Fine-tuning epochs per round
    P_GT_START         = 1.0      # Teacher-forcing prob at round 0
    P_GT_END           = 0.5      # Teacher-forcing prob at round N_ROUNDS-1
    BATCH_SIZE         = 64
    LEARNING_RATE      = 3e-4     # Lower LR for fine-tuning
    SCHED_PATIENCE     = 5
    ES_PATIENCE        = 10
    # ───────────────────────────────────────────────────────────────────────

    results: Dict = {
        "hyperparameters": {
            "n_rounds":         N_ROUNDS,
            "epochs_per_round": EPOCHS_PER_ROUND,
            "p_gt_start":       P_GT_START,
            "p_gt_end":         P_GT_END,
            "learning_rate":    LEARNING_RATE,
        },
        "teacher_forcing_schedule": [],
        "round_histories":          [],
        "val_mse_fno_input":        [],   # per round, p_gt=0
        "baseline_val_mse_fno_input": None,  # original FNO before fine-tuning
        "test_mse_fno_input":       None,
        "final_metrics":            {},
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    # ── Helper: convert raw trajectory dicts → DensityTrajectory ──────────
    def _make_trajectories(traj_list: List[Dict]) -> List[DensityTrajectory]:
        out = []
        for traj in traj_list:
            out.append(DensityTrajectory(
                time_points=np.array(traj["time_points"]),
                density_fields=np.array(traj["density_fields"]),
                delta_t=config.save_interval,
                system_params={
                    "n_particles": traj["n_particles"],
                    "masses":      traj["masses"],
                    "diameters":   traj["diameters"],
                },
            ))
        return out

    train_trajectories = _make_trajectories(dataset["train"])
    val_trajectories   = _make_trajectories(dataset["val"])
    test_trajectories  = _make_trajectories(dataset["test"])

    k = config.k_history
    in_mean  = train_dataset.input_mean
    in_std   = train_dataset.input_std
    out_mean = train_dataset.target_mean
    out_std  = train_dataset.target_std

    # ── Helper: build normalised rollout cache for a list of trajectories ─
    def _build_rollout_cache(
        model: FourierNeuralOperator,
        trajectories: List[DensityTrajectory],
    ) -> List[np.ndarray]:
        """
        Run model autoregressively over each trajectory and return a list of
        predicted frames (one array of shape (T, n_bins) per trajectory,
        in *normalised* space so the cache aligns directly with GT frames).
        """
        model.eval()
        cache = []
        with torch.no_grad():
            for traj in trajectories:
                T   = traj.n_timesteps
                gt_norm = (traj.density_fields - in_mean) / in_std  # (T, n_bins)

                # Seed with first k ground-truth frames
                seed = torch.from_numpy(
                    gt_norm[:k].astype(np.float32)
                ).unsqueeze(0).to(device)  # (1, k, n_bins)

                # Roll out for remaining T-k steps   → (T-k+1, n_bins) normalised
                n_rollout = T - k
                predicted_norm = model.predict_trajectory(seed, n_rollout)  # (T-k+1, n_bins)
                predicted_norm = predicted_norm.cpu().numpy()

                # Build aligned array of shape (T, n_bins):
                # frames 0..k-2 come from ground-truth (seed region, not predicted)
                # frames k-1..T-1 come from rollout
                rollout_arr = np.empty_like(gt_norm)
                rollout_arr[:k - 1]  = gt_norm[:k - 1]       # seed region unchanged
                rollout_arr[k - 1:]  = predicted_norm         # FNO predictions

                cache.append(rollout_arr)
        model.train()
        return cache

    # ── Inner Dataset: samples from a mix of GT and rollout frames ─────────
    class CurriculumDataset(TorchDataset):
        """
        Each sample is a (input_window, target) pair where input frames are
        drawn from ground-truth or rollout according to p_gt.

        p_gt = 1.0 → pure ground-truth (identical to Exp 3)
        p_gt = 0.0 → pure FNO rollout  (stress-test for Exp 4)
        """

        def __init__(
            self,
            trajectories: List[DensityTrajectory],
            rollout_cache: List[np.ndarray],
            p_gt: float,
        ):
            self.samples: List[Tuple[np.ndarray, np.ndarray]] = []

            for traj, rollout in zip(trajectories, rollout_cache):
                T      = traj.n_timesteps
                gt_n   = (traj.density_fields - in_mean) / in_std   # (T, n_bins)
                gt_t   = (traj.density_fields - out_mean) / out_std  # (T, n_bins)

                for t in range(k, T):
                    # Build input window of k frames ending at t-1
                    window = np.empty((k, traj.n_bins), dtype=np.float32)
                    for j in range(k):
                        frame_idx = t - k + j
                        if np.random.random() < p_gt:
                            window[j] = gt_n[frame_idx]
                        else:
                            window[j] = rollout[frame_idx]

                    target = gt_t[t].astype(np.float32)
                    self.samples.append((window, target))

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
            inp, tgt = self.samples[idx]
            return torch.from_numpy(inp), torch.from_numpy(tgt)

    # ── Helper: evaluate MSE on FNO-only inputs (p_gt=0) ──────────────────
    def _eval_fno_input_mse(
        model: FourierNeuralOperator,
        trajectories: List[DensityTrajectory],
        rollout_cache: List[np.ndarray],
    ) -> float:
        """Compute mean per-step MSE when inputs are pure FNO rollout frames."""
        model.eval()
        total_mse, total_n = 0.0, 0
        with torch.no_grad():
            for traj, rollout in zip(trajectories, rollout_cache):
                T    = traj.n_timesteps
                gt_t = (traj.density_fields - out_mean) / out_std  # (T, n_bins)

                for t in range(k, T):
                    # Window built entirely from rollout
                    window = rollout[t - k: t].astype(np.float32)  # (k, n_bins)
                    inp    = torch.from_numpy(window).unsqueeze(0).to(device)
                    pred, _ = model(inp)                            # (1, n_bins)
                    pred_np  = pred.squeeze(0).cpu().numpy()
                    mse      = float(np.mean((pred_np - gt_t[t]) ** 2))
                    total_mse += mse
                    total_n   += 1
        model.train()
        return total_mse / max(total_n, 1)

    # ── Measure baseline (original FNO) before any fine-tuning ────────────
    print("\nBuilding initial rollout caches...")
    train_rollout_cache = _build_rollout_cache(fno.to(device), train_trajectories)
    val_rollout_cache   = _build_rollout_cache(fno.to(device), val_trajectories)

    print("Measuring baseline (pre-fine-tuning) val MSE on FNO-only input...")
    baseline_val_mse = _eval_fno_input_mse(fno.to(device), val_trajectories, val_rollout_cache)
    results["baseline_val_mse_fno_input"] = float(baseline_val_mse)
    print(f"  Baseline val MSE (FNO input): {baseline_val_mse:.6f}")

    # ── Curriculum loop ────────────────────────────────────────────────────
    fno_ft = copy.deepcopy(fno).to(device)
    ft_checkpoint = os.path.join(config.output_dir, "fno_finetuned_checkpoint.pt")

    p_gt_values = np.linspace(P_GT_START, P_GT_END, N_ROUNDS).tolist()
    results["teacher_forcing_schedule"] = p_gt_values

    for rnd in range(N_ROUNDS):
        p_gt = p_gt_values[rnd]
        print(f"\n── Round {rnd + 1}/{N_ROUNDS}  (p_gt = {p_gt:.2f}) ──")

        # Rebuild dataset with current mixing ratio
        curr_train_ds = CurriculumDataset(
            train_trajectories, train_rollout_cache, p_gt=p_gt
        )
        curr_train_loader = DataLoader(
            curr_train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
        )

        # Build a pure-GT val loader (to track "standard" val loss alongside)
        val_gt_ds = CurriculumDataset(
            val_trajectories, val_rollout_cache, p_gt=1.0
        )
        val_gt_loader = DataLoader(
            val_gt_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
        )

        # Fine-tuning pass for this round
        rnd_checkpoint = os.path.join(
            config.output_dir, f"fno_ft_round_{rnd + 1}.pt"
        )
        history_rnd = train_fno(
            model                  = fno_ft,
            train_loader           = curr_train_loader,
            val_loader             = val_gt_loader,
            n_epochs               = EPOCHS_PER_ROUND,
            learning_rate          = LEARNING_RATE,
            device                 = device,
            scheduler_patience     = SCHED_PATIENCE,
            early_stopping_patience= ES_PATIENCE,
            checkpoint_path        = rnd_checkpoint,
            use_mass_conservation  = config.use_mass_conservation,
            use_positivity         = config.use_positivity,
            mass_weight            = config.mass_weight,
            positivity_weight      = config.positivity_weight,
        )

        # Load best checkpoint for this round before measuring val MSE
        if os.path.exists(rnd_checkpoint):
            ckpt = torch.load(rnd_checkpoint, map_location=device)
            fno_ft.load_state_dict(ckpt["model_state_dict"])

        # Rebuild rollout caches with the (now improved) fine-tuned model
        train_rollout_cache = _build_rollout_cache(fno_ft, train_trajectories)
        val_rollout_cache   = _build_rollout_cache(fno_ft, val_trajectories)

        # Evaluate on FNO-only input (the metric that matters)
        rnd_val_mse = _eval_fno_input_mse(fno_ft, val_trajectories, val_rollout_cache)
        results["val_mse_fno_input"].append(float(rnd_val_mse))

        results["round_histories"].append({
            "round":               rnd + 1,
            "p_gt":                p_gt,
            "train_loss_history":  [float(v) for v in history_rnd["train_loss"]],
            "val_loss_history":    [float(v) for v in history_rnd["val_loss"]],
            "val_mse_fno_input":   float(rnd_val_mse),
        })

        print(f"  Val MSE (FNO-only input): {rnd_val_mse:.6f}  "
              f"(baseline was {baseline_val_mse:.6f})")

    # ── Final evaluation on test split ─────────────────────────────────────
    print("\nBuilding test rollout cache...")
    test_rollout_cache = _build_rollout_cache(fno_ft, test_trajectories)

    print("Evaluating test MSE (FNO-only input)...")
    test_mse = _eval_fno_input_mse(fno_ft, test_trajectories, test_rollout_cache)
    results["test_mse_fno_input"] = float(test_mse)
    print(f"  Test MSE (FNO-only input):  {test_mse:.6f}")

    # Compare against baseline on val (for a clean summary)
    final_val_mse = results["val_mse_fno_input"][-1]
    improvement   = (baseline_val_mse - final_val_mse) / (baseline_val_mse + 1e-12) * 100
    results["final_metrics"] = {
        "baseline_val_mse_fno_input": float(baseline_val_mse),
        "final_val_mse_fno_input":    float(final_val_mse),
        "improvement_pct":            float(improvement),
        "test_mse_fno_input":         float(test_mse),
        "n_rounds_completed":         N_ROUNDS,
        "device_used":                device,
    }

    print("\n" + "─" * 60)
    print(f"  Baseline val MSE (FNO input) : {baseline_val_mse:.6f}")
    print(f"  Final    val MSE (FNO input) : {final_val_mse:.6f}  "
          f"({'↓' if improvement > 0 else '↑'}{abs(improvement):.1f}%)")
    print(f"  Test     MSE (FNO input)     : {test_mse:.6f}")
    print("─" * 60)

    # ── Save the fine-tuned checkpoint ─────────────────────────────────────
    save_model(fno_ft, ft_checkpoint, metadata={"config": asdict(config)})
    print(f"  Fine-tuned checkpoint saved to {ft_checkpoint}")

    return results, fno_ft


# =============================================================================
# EXPERIMENT 4: LYAPUNOV HORIZON BYPASS
# =============================================================================

def experiment_4_lyapunov_bypass(
    config: ExperimentConfig,
    fno: FourierNeuralOperator,
    dataset: Dict,
    train_dataset: DensityDataset
) -> Dict:
    """
    Demonstrate prediction beyond the Lyapunov time horizon.
    
    Figure 4: The key result figure
    - (a) MSE vs prediction horizon (in units of Lyapunov time)
    - (b) Correlation vs prediction horizon
    - (c) Example predictions at 1x, 2x, 5x Lyapunov time
    - (d) Comparison: FNO vs direct simulation with noise
    
    This is the central claim of Paper 1.
    """
    print("=" * 60)
    print("EXPERIMENT 4: Lyapunov Horizon Bypass")
    print("=" * 60)
    
    results = {
        'mse_vs_horizon': [],
        'correlation_vs_horizon': [],
        'example_predictions': [],
        'lyapunov_times': []
    }
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fno = fno.to(device)
    fno.eval()
    
    # Estimate typical Lyapunov time
    print("\nEstimating Lyapunov times for test systems...")
    
    lyapunov_times = []
    for traj in dataset['test'][:20]:
        system = ParticleSystem(config.track_length)
        for i, (m, d, v) in enumerate(zip(traj['masses'], traj['diameters'], traj['initial_velocities'])):
            system.add_particle(
                position=config.track_length * i / len(traj['masses']),
                velocity=v,
                diameter=d,
                mass=m
            )
        
        lyap = system.compute_lyapunov_exponent(perturbation=1e-8, duration=50.0)
        if lyap > 0:
            t_lyap = 1.0 / lyap
            lyapunov_times.append(t_lyap)
    
    mean_lyapunov_time = np.mean(lyapunov_times) if lyapunov_times else 10.0
    print(f"  Mean Lyapunov time: {mean_lyapunov_time:.2f} s")
    results['lyapunov_times'] = lyapunov_times
    
    # Convert test trajectories to DensityTrajectory objects
    test_trajectories = []
    for traj in dataset['test']:
        dt = DensityTrajectory(
            time_points=np.array(traj['time_points']),
            density_fields=np.array(traj['density_fields']),
            delta_t=config.save_interval,
            system_params={'n_particles': traj['n_particles']}
        )
        test_trajectories.append(dt)
    
    # Prediction at various horizons
    print("\nEvaluating prediction at various horizons...")
    
    horizons_in_lyapunov_units = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
    
    for horizon_factor in horizons_in_lyapunov_units:
        horizon_time = horizon_factor * mean_lyapunov_time
        n_steps = int(horizon_time / config.save_interval)
        
        mse_samples = []
        corr_samples = []
        
        for traj in test_trajectories[:20]:
            if n_steps < traj.n_timesteps:
                # Build k-frame seed window from the start of the trajectory
                k = config.k_history
                seed_frames = traj.density_fields[:k]  # (k, n_bins)
                seed_norm = (seed_frames - train_dataset.input_mean) / train_dataset.input_std
                seed_tensor = torch.from_numpy(seed_norm.astype(np.float32)).to(device)  # (k, n_bins)

                # Predict trajectory (returns (n_rollout+1, n_bins) starting at frame k-1)
                n_rollout = traj.n_timesteps - k
                with torch.no_grad():
                    predicted_norm = fno.predict_trajectory(seed_tensor, n_rollout)
                    predicted = train_dataset.denormalize_output(predicted_norm).cpu().numpy()

                # Align predicted[n_steps] with ground-truth frame k-1+n_steps
                true_idx = (k - 1) + n_steps
                if n_steps >= len(predicted) or true_idx >= traj.n_timesteps:
                    continue  # skip — horizon exceeds available data for this trajectory

                pred_final = predicted[n_steps]
                true_final = traj.density_fields[true_idx]

                # Compute metrics
                mse = np.mean((pred_final - true_final) ** 2)

                # Correlation
                pred_flat = pred_final.flatten()
                true_flat = true_final.flatten()
                if np.std(pred_flat) > 0 and np.std(true_flat) > 0:
                    corr = np.corrcoef(pred_flat, true_flat)[0, 1]
                else:
                    corr = 0.0

                mse_samples.append(mse)
                corr_samples.append(corr)
        
        if mse_samples:
            results['mse_vs_horizon'].append({
                'horizon_lyapunov_units': horizon_factor,
                'horizon_time': horizon_time,
                'mse_mean': float(np.mean(mse_samples)),
                'mse_std': float(np.std(mse_samples)),
                'n_samples': len(mse_samples)
            })
            results['correlation_vs_horizon'].append({
                'horizon_lyapunov_units': horizon_factor,
                'correlation_mean': float(np.mean(corr_samples)),
                'correlation_std': float(np.std(corr_samples))
            })
            print(f"  {horizon_factor}x Lyapunov: MSE = {np.mean(mse_samples):.6f}, Corr = {np.mean(corr_samples):.4f}")
    
    # Example predictions for visualization
    print("\nGenerating example predictions...")

    example_traj = test_trajectories[0]
    k = config.k_history
    seed_frames = example_traj.density_fields[:k]
    seed_norm = (seed_frames - train_dataset.input_mean) / train_dataset.input_std
    seed_tensor = torch.from_numpy(seed_norm.astype(np.float32)).to(device)

    for horizon_factor in [1.0, 2.0, 5.0]:
        horizon_time = horizon_factor * mean_lyapunov_time
        n_steps = min(int(horizon_time / config.save_interval), example_traj.n_timesteps - 1)
        n_rollout = example_traj.n_timesteps - k

        with torch.no_grad():
            predicted_norm = fno.predict_trajectory(seed_tensor, min(n_steps, n_rollout))
            predicted = train_dataset.denormalize_output(predicted_norm).cpu().numpy()

        pred_step = min(n_steps, len(predicted) - 1)
        true_step = min((k - 1) + n_steps, example_traj.n_timesteps - 1)
        pred_final = predicted[pred_step]

        results['example_predictions'].append({
            'horizon_lyapunov_units': horizon_factor,
            'n_steps': n_steps,
            'initial_density': example_traj.density_fields[k - 1].tolist(),
            'predicted_density': pred_final.tolist(),
            'true_density': example_traj.density_fields[true_step].tolist()
        })
    
    return results



# =============================================================================
# EXPERIMENT 5: ABLATION STUDIES
# =============================================================================

def experiment_5_ablations(config: ExperimentConfig, dataset: Dict, train_dataset: DensityDataset) -> Dict:
    """
    Ablation studies to understand what drives performance.
    
    Each variant is trained to convergence before evaluation so that
    differences in validation loss reflect true capacity/inductive-bias
    differences rather than random initialisation.
    
    Table 2: Ablation results
    - Different numbers of Fourier modes
    - Different layer depths
    - Different hidden channel widths
    
    Figure 5: Ablation visualizations
    """
    print("=" * 60)
    print("EXPERIMENT 5: Ablation Studies")
    print("=" * 60)
    
    results = {
        'ablations': []
    }
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    # -------------------------------------------------------------------------
    # Build training DataLoader from train_dataset (normalization already fit)
    # -------------------------------------------------------------------------
    train_loader_abl = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=0
    )
    
    # Build validation DataLoader with the *same* normalization as training
    val_trajectories = []
    for traj in dataset['val']:
        dt = DensityTrajectory(
            time_points=np.array(traj['time_points']),
            density_fields=np.array(traj['density_fields']),
            delta_t=config.save_interval
        )
        val_trajectories.append(dt)
    
    # Re-use train_dataset stats so ablation variants see identical normalisation
    val_dataset_abl = DensityDataset(val_trajectories, n_steps_ahead=1, normalize=True,
                                     k_history=config.k_history)
    val_dataset_abl.input_mean  = train_dataset.input_mean
    val_dataset_abl.input_std   = train_dataset.input_std
    val_dataset_abl.target_mean = train_dataset.target_mean
    val_dataset_abl.target_std  = train_dataset.target_std
    val_loader_abl = DataLoader(val_dataset_abl, batch_size=64, shuffle=False, num_workers=0)
    
    # Default ablation values must match ExperimentConfig (n_modes=32, hidden_channels=64, n_layers=4).
    # Each variant changes exactly ONE axis so that differences isolate a single factor.
    default_modes    = config.fno_modes            # 32
    default_channels = config.fno_hidden_channels  # 64
    default_layers   = config.fno_layers           # 4

    ablation_configs = [
        {'name': 'Full model (default)',  'n_modes': default_modes,    'hidden_channels': default_channels, 'n_layers': default_layers},
        {'name': 'Fewer modes (8)',       'n_modes': 8,                'hidden_channels': default_channels, 'n_layers': default_layers},
        {'name': 'More modes (48)',       'n_modes': 48,               'hidden_channels': default_channels, 'n_layers': default_layers},
        {'name': 'Shallow (2 layers)',    'n_modes': default_modes,    'hidden_channels': default_channels, 'n_layers': 2},
        {'name': 'Deep (6 layers)',       'n_modes': default_modes,    'hidden_channels': default_channels, 'n_layers': 6},
        {'name': 'Narrow (32 channels)',  'n_modes': default_modes,    'hidden_channels': 32,               'n_layers': default_layers},
        {'name': 'Wide (128 channels)',   'n_modes': default_modes,    'hidden_channels': 128,              'n_layers': default_layers},
    ]
    
    for abl in ablation_configs:
        print(f"\n  Training variant: {abl['name']}...")
        
        fno_abl = FourierNeuralOperator(
            n_bins=config.n_density_bins,
            n_modes=abl['n_modes'],
            hidden_channels=abl['hidden_channels'],
            n_layers=abl['n_layers'],
            delta_t=config.save_interval,
            k_history=config.k_history,
            lstm_hidden=config.lstm_hidden,
            lstm_layers=config.lstm_layers
        )
        
        # Train each variant to convergence.
        # 50 epochs with early-stopping (patience 10) keeps ablations tractable
        # while ensuring a fair comparison between architectures.
        abl_checkpoint = os.path.join(
            config.output_dir,
            f"ablation_{abl['name'].replace(' ', '_').replace('(', '').replace(')', '')}.pt"
        )
        history_abl = train_fno(
            model=fno_abl,
            train_loader=train_loader_abl,
            val_loader=val_loader_abl,
            n_epochs=50,
            learning_rate=1e-3,
            device=device,
            scheduler_patience=5,
            early_stopping_patience=10,
            checkpoint_path=abl_checkpoint
        )
        
        # Load best checkpoint for evaluation
        if os.path.exists(abl_checkpoint):
            checkpoint = torch.load(abl_checkpoint, map_location=device)
            fno_abl.load_state_dict(checkpoint['model_state_dict'])
        
        # Evaluate on the full validation set
        fno_abl.eval()
        val_losses = []
        with torch.no_grad():
            for inputs, targets in val_loader_abl:
                inputs, targets = inputs.to(device), targets.to(device)
                loss = fno_abl.compute_loss(inputs, targets)
                val_losses.append(loss.item())
        
        best_val = min(history_abl['val_loss'])
        final_val = float(np.mean(val_losses))
        
        results['ablations'].append({
            'name': abl['name'],
            'config': abl,
            'best_val_loss': best_val,
            'final_val_loss': final_val,
            'n_epochs_trained': len(history_abl['train_loss']),
            'training_history': [
                {
                    'epoch': i,
                    'train_loss': float(history_abl['train_loss'][i]),
                    'val_loss':   float(history_abl['val_loss'][i])
                }
                for i in range(len(history_abl['train_loss']))
            ],
            'n_parameters': sum(p.numel() for p in fno_abl.parameters())
        })
        print(f"    Best Val Loss: {best_val:.6f}, Epochs: {len(history_abl['train_loss'])}, "
              f"Params: {results['ablations'][-1]['n_parameters']:,}")
    
    # Print summary table
    print("\n" + "-" * 60)
    print(f"{'Variant':<25} {'Best Val Loss':<15} {'Epochs':<8} {'Params':>10}")
    print("-" * 60)
    for abl_res in results['ablations']:
        print(f"{abl_res['name']:<25} {abl_res['best_val_loss']:<15.6f} "
              f"{abl_res['n_epochs_trained']:<8} {abl_res['n_parameters']:>10,}")
    
    return results


# =============================================================================
# EXPERIMENT 6: BASELINE COMPARISONS
# =============================================================================

def experiment_6_baselines(config: ExperimentConfig, dataset: Dict) -> Dict:
    """
    Compare FNO against baseline methods.
    
    Table 3: Method comparison
    - FNO (ours) - from Experiment 3
    - Persistence baseline (ρ(t+Δt) = ρ(t))
    - Linear extrapolation
    - Vanilla MLP
    - LSTM/RNN (Methods lines 413-421)
    - U-Net CNN (Methods lines 423-430)
    - EDMD (Methods lines 401-407)
    - Ulam's method (Methods lines 409-411)
    
    Figure 6: Comparison plots
    """
    print("=" * 60)
    print("EXPERIMENT 6: Baseline Comparisons")
    print("=" * 60)
    
    results = {
        'baselines': []
    }
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    # Prepare single-frame data (for simple baselines: persistence, mean, linear, MLP)
    def prepare_data(trajectories):
        inputs, targets = [], []
        for traj in trajectories:
            df = np.array(traj['density_fields'])
            for t in range(len(df) - 1):
                inputs.append(df[t])
                targets.append(df[t + 1])
        return np.array(inputs), np.array(targets)

    # Prepare k-frame windowed data (for LSTM and U-Net — matches FNO input format)
    def prepare_windowed_data(trajectories, k_history: int = 4):
        inputs, targets = [], []
        for traj in trajectories:
            df = np.array(traj['density_fields'])
            for t in range(len(df) - k_history):
                inputs.append(df[t: t + k_history])          # (k_history, n_bins)
                targets.append(df[t + k_history])             # (n_bins,)
        return np.array(inputs), np.array(targets)

    k_history = config.k_history

    train_inputs, train_targets = prepare_data(dataset['train'])
    val_inputs, val_targets = prepare_data(dataset['val'])

    train_inputs_w, train_targets_w = prepare_windowed_data(dataset['train'], k_history)
    val_inputs_w, val_targets_w = prepare_windowed_data(dataset['val'], k_history)

    print(f"\n  Single-frame training samples: {len(train_inputs)}")
    print(f"  Single-frame validation samples: {len(val_inputs)}")
    print(f"  Windowed (k={k_history}) training samples: {len(train_inputs_w)}")
    print(f"  Windowed (k={k_history}) validation samples: {len(val_inputs_w)}")
    
    # =========================================================================
    # Simple Baselines (no training required)
    # =========================================================================
    
    # Baseline 1: Persistence (predict same as input)
    print("\n  [1/8] Persistence baseline...")
    persistence_preds = val_inputs.copy()
    persistence_mse = np.mean((persistence_preds - val_targets) ** 2)
    results['baselines'].append({
        'name': 'Persistence',
        'mse': float(persistence_mse),
        'description': 'ρ(t+Δt) = ρ(t)',
        'type': 'simple'
    })
    print(f"       MSE: {persistence_mse:.6f}")
    
    # Baseline 2: Global mean
    print("\n  [2/8] Global mean baseline...")
    global_mean = np.mean(val_inputs, axis=0)
    mean_preds = np.tile(global_mean, (len(val_inputs), 1))
    mean_mse = np.mean((mean_preds - val_targets) ** 2)
    results['baselines'].append({
        'name': 'Global Mean',
        'mse': float(mean_mse),
        'description': 'ρ(t+Δt) = mean(ρ)',
        'type': 'simple'
    })
    print(f"       MSE: {mean_mse:.6f}")
    
    # Baseline 3: Linear extrapolation
    print("\n  [3/8] Linear extrapolation baseline...")
    linear_mses = []
    for traj in dataset['val']:
        df = np.array(traj['density_fields'])
        for t in range(1, len(df) - 1):
            pred = 2 * df[t] - df[t-1]
            mse = np.mean((pred - df[t+1]) ** 2)
            linear_mses.append(mse)
    linear_mse = np.mean(linear_mses)
    results['baselines'].append({
        'name': 'Linear Extrapolation',
        'mse': float(linear_mse),
        'description': 'ρ(t+Δt) = 2ρ(t) - ρ(t-Δt)',
        'type': 'simple'
    })
    print(f"       MSE: {linear_mse:.6f}")
    
    # Baseline 4: Untrained MLP
    print("\n  [4/8] Simple MLP baseline...")
    mlp_weights = np.random.randn(config.n_density_bins, config.n_density_bins) / config.n_density_bins
    mlp_preds = np.tanh(val_inputs @ mlp_weights)
    mlp_mse = np.mean((mlp_preds - val_targets) ** 2)
    results['baselines'].append({
        'name': 'MLP (untrained)',
        'mse': float(mlp_mse),
        'description': 'Random single-layer MLP',
        'type': 'simple'
    })
    print(f"       MSE: {mlp_mse:.6f}")
    
    # =========================================================================
    # Neural Network Baselines (require training)
    # =========================================================================
    #
    # LSTM and U-Net receive the same k-frame windowed input as the FNO so
    # that all learned baselines have identical temporal context.  This is a
    # fair comparison: same information, different architectures.
    # =========================================================================

    # Convert to PyTorch datasets
    def create_torch_datasets(train_in, train_tgt, val_in, val_tgt):
        train_dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(train_in).float(),
            torch.from_numpy(train_tgt).float()
        )
        val_dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(val_in).float(),
            torch.from_numpy(val_tgt).float()
        )
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        return train_loader, val_loader

    # Windowed loaders for LSTM and U-Net (shape: batch, k_history, n_bins)
    train_loader_w, val_loader_w = create_torch_datasets(
        train_inputs_w, train_targets_w, val_inputs_w, val_targets_w
    )

    # Baseline 5: LSTM (k-frame windowed input)
    print(f"\n  [5/8] LSTM baseline (k={k_history} frame input)...")
    lstm = LSTMDensityPredictor(
        n_bins=config.n_density_bins,
        hidden_dim=256,
        n_layers=2,
        k_history=k_history
    )
    history_lstm = train_baseline_model(
        lstm, train_loader_w, val_loader_w,
        n_epochs=50, learning_rate=1e-3, device=device
    )

    lstm.eval()
    lstm_losses = []
    with torch.no_grad():
        for inputs, targets in val_loader_w:
            inputs, targets = inputs.to(device), targets.to(device)
            preds = lstm(inputs)
            loss = F.mse_loss(preds, targets)
            lstm_losses.append(loss.item())
    lstm_mse = np.mean(lstm_losses)

    results['baselines'].append({
        'name': 'LSTM',
        'mse': float(lstm_mse),
        'description': f'2-layer LSTM, 256 hidden, k={k_history} frame input',
        'type': 'neural',
        'n_parameters': sum(p.numel() for p in lstm.parameters())
    })
    print(f"       MSE: {lstm_mse:.6f}, Params: {results['baselines'][-1]['n_parameters']:,}")

    # Baseline 6: U-Net (k-frame windowed input as multi-channel)
    print(f"\n  [6/8] U-Net baseline (k={k_history} channel input)...")
    unet = UNet1D(n_bins=config.n_density_bins, base_channels=32,
                  in_channels=k_history)
    history_unet = train_baseline_model(
        unet, train_loader_w, val_loader_w,
        n_epochs=50, learning_rate=1e-3, device=device
    )

    unet.eval()
    unet_losses = []
    with torch.no_grad():
        for inputs, targets in val_loader_w:
            inputs, targets = inputs.to(device), targets.to(device)
            preds = unet(inputs)
            loss = F.mse_loss(preds, targets)
            unet_losses.append(loss.item())
    unet_mse = np.mean(unet_losses)
    
    results['baselines'].append({
        'name': 'U-Net',
        'mse': float(unet_mse),
        'description': f'1D U-Net with skip connections, k={k_history} channel input',
        'type': 'neural',
        'n_parameters': sum(p.numel() for p in unet.parameters())
    })
    print(f"       MSE: {unet_mse:.6f}, Params: {results['baselines'][-1]['n_parameters']:,}")
    
    # =========================================================================
    # Classical Operator Approximation Methods
    # =========================================================================
    
    # Baseline 7: EDMD (Extended DMD)
    print("\n  [7/8] EDMD baseline...")
    edmd = EDMDPredictor(n_modes=32, reg_param=1e-6)
    edmd.fit(train_inputs, train_targets)
    
    edmd_preds = edmd.predict(val_inputs)
    edmd_mse = np.mean((edmd_preds - val_targets) ** 2)
    
    results['baselines'].append({
        'name': 'EDMD',
        'mse': float(edmd_mse),
        'description': 'Extended DMD with Fourier basis',
        'type': 'classical',
        'n_modes': 32
    })
    print(f"       MSE: {edmd_mse:.6f}")
    
    # Baseline 8: Ulam's Method
    print("\n  [8/8] Ulam's method baseline...")
    ulam = UlamPredictor(n_bins=config.n_density_bins)
    
    # Prepare trajectories for Ulam (needs full trajectories, not pairs)
    train_trajectories = [np.array(traj['density_fields']) for traj in dataset['train']]
    ulam.fit(train_trajectories)
    
    ulam_preds = ulam.predict(val_inputs)
    ulam_mse = np.mean((ulam_preds - val_targets) ** 2)
    
    results['baselines'].append({
        'name': "Ulam's Method",
        'mse': float(ulam_mse),
        'description': 'Discretized Perron-Frobenius operator',
        'type': 'classical',
        'n_bins': config.n_density_bins
    })
    print(f"       MSE: {ulam_mse:.6f}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    
    print("\n" + "=" * 60)
    print("BASELINE COMPARISON SUMMARY")
    print("=" * 60)
    print(f"\n{'Method':<25} {'MSE':<12} {'Type':<10}")
    print("-" * 50)
    
    for baseline in results['baselines']:
        print(f"{baseline['name']:<25} {baseline['mse']:<12.6f} {baseline['type']:<10}")
    
    print("\nNote: FNO results available from Experiment 3")
    
    return results



# =============================================================================
# EXPERIMENT 7: INVERSE PROBLEM
# =============================================================================

def experiment_7_inverse_problem(config: ExperimentConfig, dataset: Dict) -> Dict:
    """
    Inverse problem: infer masses from collision data.
    
    The CollisionGraphEncoder is trained on a training split of collision
    sequences (built from the training trajectories) before evaluation on
    a held-out test split, so reported errors reflect learned inference.
    
    Figure 7: Inverse problem results
    - (a) Mass inference accuracy vs observation time
    - (b) Confusion matrix for mass categories
    - (c) Effect of collision count on accuracy
    
    This demonstrates the novel "shadow" inverse problem.
    """
    print("=" * 60)
    print("EXPERIMENT 7: Inverse Problem")
    print("=" * 60)
    
    results = {
        'encoder_training': {},
        'mass_inference': [],
        'diameter_inference': [],
        'vs_observation_time': [],
        'vs_collision_count': []
    }
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    # -------------------------------------------------------------------------
    # Helper: build a CollisionSequence from a trajectory dict
    # -------------------------------------------------------------------------
    def build_collision_sequence(traj: Dict, sim_duration: float) -> Optional[CollisionSequence]:
        n_particles = traj['n_particles']
        system = ParticleSystem(config.track_length)
        spacing = config.track_length / n_particles
        for i in range(n_particles):
            system.add_particle(
                position=spacing * i + spacing * 0.5,
                velocity=traj['initial_velocities'][i],
                diameter=traj['diameters'][i],
                mass=traj['masses'][i]
            )
        _, collisions = system.evolve(duration=sim_duration, save_interval=1.0)
        if len(collisions) <= 5:
            return None
        return CollisionSequence(
            times=np.array([c.time for c in collisions]),
            particle_pairs=np.array([[c.particle_i, c.particle_j] for c in collisions]),
            momentum_transfers=np.array([c.delta_p for c in collisions]),
            positions=np.array([c.position for c in collisions]),
            true_masses=np.array(traj['masses']),
            true_diameters=np.array(traj['diameters'])
        )
    
    # -------------------------------------------------------------------------
    # Build training collision sequences (from training set trajectories)
    # -------------------------------------------------------------------------
    print("\nBuilding training collision sequences (from training set)...")
    train_collision_seqs = []
    # Cap at 50 training trajectories to keep encoder training tractable
    for traj in dataset['train'][:50]:
        seq = build_collision_sequence(traj, sim_duration=config.simulation_time)
        if seq is not None:
            train_collision_seqs.append(seq)
    print(f"  Training sequences: {len(train_collision_seqs)}")
    
    # -------------------------------------------------------------------------
    # Build test collision sequences (from test set trajectories)
    # -------------------------------------------------------------------------
    print("\nBuilding test collision sequences (from test set)...")
    test_collision_seqs = []
    for traj in dataset['test'][:30]:
        seq = build_collision_sequence(traj, sim_duration=config.simulation_time)
        if seq is not None:
            test_collision_seqs.append(seq)
    print(f"  Test sequences: {len(test_collision_seqs)}")
    
    if len(train_collision_seqs) == 0:
        print("  WARNING: No training collision sequences available. Skipping Experiment 7.")
        return results
    
    # -------------------------------------------------------------------------
    # Determine a representative particle count for the shared encoder.
    # We use the mode (most common n_particles) across all sequences so that
    # the single shared encoder covers the majority of systems.
    # -------------------------------------------------------------------------
    all_seqs = train_collision_seqs + test_collision_seqs
    n_particles_counts = [len(s.true_masses) for s in all_seqs]
    representative_n = int(np.median(n_particles_counts))
    print(f"\n  Representative n_particles for encoder: {representative_n}")
    
    # Filter to only sequences matching the representative particle count so
    # the graph encoder dimensions stay consistent during training.
    train_seqs_filtered = [s for s in train_collision_seqs if len(s.true_masses) == representative_n]
    test_seqs_filtered  = [s for s in test_collision_seqs  if len(s.true_masses) == representative_n]
    
    if len(train_seqs_filtered) == 0:
        # Fall back to the most common n_particles value
        from collections import Counter
        representative_n = Counter(n_particles_counts).most_common(1)[0][0]
        train_seqs_filtered = [s for s in train_collision_seqs if len(s.true_masses) == representative_n]
        test_seqs_filtered  = [s for s in test_collision_seqs  if len(s.true_masses) == representative_n]
    
    print(f"  Filtered training sequences: {len(train_seqs_filtered)}")
    print(f"  Filtered test sequences: {len(test_seqs_filtered)}")
    
    # -------------------------------------------------------------------------
    # Train the CollisionGraphEncoder on the training split
    # -------------------------------------------------------------------------
    print("\nTraining CollisionGraphEncoder...")
    
    encoder = CollisionGraphEncoder(
        n_particles=representative_n,
        embedding_dim=64
    )
    
    encoder_history = train_collision_encoder(
        model=encoder,
        collision_sequences=train_seqs_filtered,
        n_epochs=100,
        learning_rate=1e-3,
        device=device
    )
    
    encoder = encoder.to(device)
    encoder.eval()
    
    results['encoder_training'] = {
        'n_train_sequences': len(train_seqs_filtered),
        'n_epochs': len(encoder_history['loss']),
        'final_loss': float(encoder_history['loss'][-1]),
        'best_loss': float(min(encoder_history['loss'])),
        'loss_history': [float(v) for v in encoder_history['loss']]
    }
    print(f"  Encoder training complete. Best loss: {results['encoder_training']['best_loss']:.6f}")
    
    # -------------------------------------------------------------------------
    # Evaluate on the held-out test split
    # -------------------------------------------------------------------------
    print("\nEvaluating mass/diameter inference on test sequences...")
    
    for seq in test_seqs_filtered:
        times_t = torch.from_numpy(seq.times).float().to(device)
        momentum_t = torch.from_numpy(seq.momentum_transfers).float().to(device)
        positions_t = torch.from_numpy(seq.positions).float().to(device)
        pairs_t = torch.from_numpy(seq.particle_pairs).long().to(device)
        
        with torch.no_grad():
            predictions = encoder(times_t, momentum_t, positions_t, pairs_t)
        
        pred_masses    = predictions['masses'].cpu().numpy()
        pred_diameters = predictions['diameters'].cpu().numpy()
        
        mass_error     = float(np.mean(np.abs(pred_masses    - seq.true_masses)    / seq.true_masses))
        diameter_error = float(np.mean(np.abs(pred_diameters - seq.true_diameters) / seq.true_diameters))
        
        results['mass_inference'].append({
            'n_particles': len(seq.true_masses),
            'n_collisions': seq.n_collisions,
            'relative_mass_error': mass_error,
            'relative_diameter_error': diameter_error
        })
    
    if results['mass_inference']:
        mean_mass_err = np.mean([r['relative_mass_error'] for r in results['mass_inference']])
        mean_diam_err = np.mean([r['relative_diameter_error'] for r in results['mass_inference']])
        print(f"  Mean relative mass error:     {mean_mass_err:.4f}")
        print(f"  Mean relative diameter error: {mean_diam_err:.4f}")
    
    # -------------------------------------------------------------------------
    # Accuracy vs observation time
    # Using the same trained encoder, truncate sequences to each window
    # -------------------------------------------------------------------------
    print("\nAccuracy vs observation time (trained encoder)...")
    
    observation_times = [10, 25, 50, 75, 100]
    
    for obs_time in observation_times:
        errors = []
        
        for seq in test_seqs_filtered:
            mask = seq.times <= obs_time
            if np.sum(mask) < 3:
                continue
            
            times_t    = torch.from_numpy(seq.times[mask]).float().to(device)
            momentum_t = torch.from_numpy(seq.momentum_transfers[mask]).float().to(device)
            positions_t= torch.from_numpy(seq.positions[mask]).float().to(device)
            pairs_t    = torch.from_numpy(seq.particle_pairs[mask]).long().to(device)
            
            with torch.no_grad():
                preds = encoder(times_t, momentum_t, positions_t, pairs_t)
            
            pred_masses = preds['masses'].cpu().numpy()
            error = float(np.mean(np.abs(pred_masses - seq.true_masses) / seq.true_masses))
            errors.append(error)
        
        if errors:
            results['vs_observation_time'].append({
                'observation_time': obs_time,
                'n_sequences': len(errors),
                'mean_error': float(np.mean(errors)),
                'std_error': float(np.std(errors))
            })
            print(f"  t_obs={obs_time:3d}s: Error = {np.mean(errors):.4f} ± {np.std(errors):.4f} "
                  f"(n={len(errors)})")
    
    return results


# =============================================================================
# EXPERIMENT 8: GRID CONVERGENCE STUDY
# =============================================================================

def experiment_8_grid_convergence(
    config: ExperimentConfig,
    dataset: Dict,
) -> Dict:
    """
    Grid convergence study: train and evaluate at multiple spatial resolutions.

    JCP reviewers expect to see that the learned operator converges as the
    discretisation is refined.  We re-bin the *same* trajectory data at
    n ∈ {64, 128, 256, 512} and train separate FNO instances at each
    resolution, reporting single-step MSE and multi-step rollout error.

    Table / Figure: relative L2 error vs spatial resolution n.
    """
    print("=" * 60)
    print("EXPERIMENT 8: Grid Convergence Study")
    print("=" * 60)

    resolutions = [64, 128, 256, 512]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")

    results = {
        'resolutions': [],
    }

    # ── Helper: re-bin a trajectory's density fields to a new resolution ──
    def rebin_trajectory(traj: Dict, n_bins_new: int) -> Dict:
        """Resample density fields from their original resolution to n_bins_new."""
        import copy
        traj_new = copy.deepcopy(traj)
        df = np.array(traj['density_fields'])            # (T, n_bins_orig)
        n_bins_orig = df.shape[1]

        if n_bins_new == n_bins_orig:
            return traj_new

        # Linear interpolation to new grid
        x_orig = np.linspace(0, 1, n_bins_orig)
        x_new  = np.linspace(0, 1, n_bins_new)
        rebinned = np.array([np.interp(x_new, x_orig, row) for row in df])

        # Rescale so that total mass (sum * bin_width) is conserved
        scale = n_bins_orig / n_bins_new
        rebinned *= scale

        traj_new['density_fields'] = rebinned.tolist()
        return traj_new

    for n_bins in resolutions:
        print(f"\n── Resolution n = {n_bins} ──")

        # Re-bin dataset
        print("  Re-binning dataset...")
        ds_rebinned = {
            split: [rebin_trajectory(t, n_bins) for t in dataset[split]]
            for split in ('train', 'val', 'test')
        }

        # Build DensityTrajectory objects
        def to_dt(traj_list):
            return [
                DensityTrajectory(
                    time_points=np.array(t['time_points']),
                    density_fields=np.array(t['density_fields']),
                    delta_t=config.save_interval,
                )
                for t in traj_list
            ]

        train_traj = to_dt(ds_rebinned['train'])
        val_traj   = to_dt(ds_rebinned['val'])
        test_traj  = to_dt(ds_rebinned['test'])

        train_ds = DensityDataset(train_traj, n_steps_ahead=1, normalize=True,
                                  k_history=config.k_history)
        val_ds   = DensityDataset(val_traj,   n_steps_ahead=1, normalize=True,
                                  k_history=config.k_history)
        val_ds.input_mean  = train_ds.input_mean
        val_ds.input_std   = train_ds.input_std
        val_ds.target_mean = train_ds.target_mean
        val_ds.target_std  = train_ds.target_std

        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
        val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False, num_workers=0)

        # Keep n_modes <= n_bins // 2 (Nyquist limit)
        n_modes = min(config.fno_modes, n_bins // 2)

        print(f"  Training FNO (modes={n_modes}, samples={len(train_ds)})...")
        fno_res = FourierNeuralOperator(
            n_bins=n_bins,
            n_modes=n_modes,
            hidden_channels=config.fno_hidden_channels,
            n_layers=config.fno_layers,
            delta_t=config.save_interval,
            k_history=config.k_history,
            lstm_hidden=config.lstm_hidden,
            lstm_layers=config.lstm_layers,
        )

        ckpt_path = os.path.join(config.output_dir, f"fno_grid_n{n_bins}.pt")
        history = train_fno(
            model=fno_res,
            train_loader=train_loader,
            val_loader=val_loader,
            n_epochs=50,
            learning_rate=1e-3,
            device=device,
            scheduler_patience=5,
            early_stopping_patience=10,
            checkpoint_path=ckpt_path,
        )

        # Load best checkpoint
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device)
            fno_res.load_state_dict(ckpt['model_state_dict'])

        # Evaluate single-step MSE on validation set
        fno_res.eval()
        val_losses = []
        with torch.no_grad():
            for inp, tgt in val_loader:
                inp, tgt = inp.to(device), tgt.to(device)
                loss = fno_res.compute_loss(inp, tgt)
                val_losses.append(loss.item())
        single_step_mse = float(np.mean(val_losses))

        # Evaluate multi-step rollout (10 steps) on first 10 test trajectories
        rollout_mses = []
        n_rollout_steps = 10
        fno_res.eval()
        with torch.no_grad():
            for traj in test_traj[:10]:
                k = config.k_history
                if traj.n_timesteps < k + n_rollout_steps:
                    continue
                seed = traj.density_fields[:k]
                seed_norm = (seed - train_ds.input_mean) / train_ds.input_std
                seed_t = torch.from_numpy(seed_norm.astype(np.float32)).to(device)

                pred_norm = fno_res.predict_trajectory(seed_t, n_rollout_steps)
                pred = train_ds.denormalize_output(pred_norm).cpu().numpy()

                true_idx = k - 1 + n_rollout_steps
                if true_idx < traj.n_timesteps and n_rollout_steps < len(pred):
                    mse = float(np.mean((pred[n_rollout_steps] - traj.density_fields[true_idx]) ** 2))
                    rollout_mses.append(mse)

        rollout_mse = float(np.mean(rollout_mses)) if rollout_mses else float('nan')

        entry = {
            'n_bins': n_bins,
            'n_modes': n_modes,
            'single_step_mse': single_step_mse,
            'best_val_loss': float(min(history['val_loss'])),
            'rollout_10step_mse': rollout_mse,
            'n_parameters': sum(p.numel() for p in fno_res.parameters()),
            'n_epochs_trained': len(history['train_loss']),
        }
        results['resolutions'].append(entry)

        print(f"  Single-step MSE: {single_step_mse:.6f}")
        print(f"  10-step rollout MSE: {rollout_mse:.6f}")
        print(f"  Parameters: {entry['n_parameters']:,}")

    # Summary table
    print("\n" + "=" * 70)
    print("GRID CONVERGENCE SUMMARY")
    print("=" * 70)
    print(f"  {'n_bins':>6}  {'1-step MSE':>12}  {'10-step MSE':>12}  {'Params':>10}")
    print("  " + "-" * 46)
    for r in results['resolutions']:
        print(f"  {r['n_bins']:>6}  {r['single_step_mse']:>12.6f}  "
              f"{r['rollout_10step_mse']:>12.6f}  {r['n_parameters']:>10,}")

    return results


# =============================================================================
# EXPERIMENT 9: MULTI-SEED TRAINING (Statistical Confidence)
# =============================================================================

def experiment_9_multi_seed(
    config: ExperimentConfig,
    dataset: Dict,
    n_seeds: int = 3,
) -> Dict:
    """
    Train the FNO with multiple random seeds to obtain confidence intervals.

    This addresses the methods claim (line 474) about independent training runs.
    We train n_seeds models from scratch with different initialisations and
    report mean ± std of validation loss and multi-step rollout metrics.

    Table: per-seed val loss, mean, std.
    """
    print("=" * 60)
    print(f"EXPERIMENT 9: Multi-Seed Training ({n_seeds} seeds)")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")

    results = {
        'n_seeds': n_seeds,
        'seeds': [],
        'summary': {},
    }

    # Build data loaders (shared across all seeds — only model init changes)
    def to_dt(traj_list):
        return [
            DensityTrajectory(
                time_points=np.array(t['time_points']),
                density_fields=np.array(t['density_fields']),
                delta_t=config.save_interval,
            )
            for t in traj_list
        ]

    train_traj = to_dt(dataset['train'])
    val_traj   = to_dt(dataset['val'])
    test_traj  = to_dt(dataset['test'])

    train_ds = DensityDataset(train_traj, n_steps_ahead=1, normalize=True,
                              k_history=config.k_history)
    val_ds   = DensityDataset(val_traj,   n_steps_ahead=1, normalize=True,
                              k_history=config.k_history)
    val_ds.input_mean  = train_ds.input_mean
    val_ds.input_std   = train_ds.input_std
    val_ds.target_mean = train_ds.target_mean
    val_ds.target_std  = train_ds.target_std

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False, num_workers=0)

    best_val_losses = []
    rollout_mses_all = []

    for seed_idx in range(n_seeds):
        model_seed = config.seed + seed_idx * 1000
        print(f"\n── Seed {seed_idx + 1}/{n_seeds}  (torch seed = {model_seed}) ──")

        torch.manual_seed(model_seed)
        np.random.seed(model_seed)

        fno_s = FourierNeuralOperator(
            n_bins=config.n_density_bins,
            n_modes=config.fno_modes,
            hidden_channels=config.fno_hidden_channels,
            n_layers=config.fno_layers,
            delta_t=config.save_interval,
            k_history=config.k_history,
            lstm_hidden=config.lstm_hidden,
            lstm_layers=config.lstm_layers,
        )

        ckpt_path = os.path.join(config.output_dir, f"fno_seed{seed_idx}.pt")
        history = train_fno(
            model=fno_s,
            train_loader=train_loader,
            val_loader=val_loader,
            n_epochs=100,
            learning_rate=1e-3,
            device=device,
            scheduler_patience=10,
            early_stopping_patience=20,
            checkpoint_path=ckpt_path,
        )

        # Load best checkpoint
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device)
            fno_s.load_state_dict(ckpt['model_state_dict'])

        best_val = float(min(history['val_loss']))
        best_val_losses.append(best_val)

        # Multi-step rollout evaluation on test set
        fno_s.eval()
        k = config.k_history
        horizon_factors = [1.0, 2.0, 5.0, 10.0]

        # Compute Lyapunov times from test set
        lyapunov_times = []
        for traj in dataset['test'][:10]:
            system = ParticleSystem(config.track_length)
            for i, (m, d, v) in enumerate(zip(traj['masses'], traj['diameters'], traj['initial_velocities'])):
                system.add_particle(
                    position=config.track_length * i / len(traj['masses']),
                    velocity=v, diameter=d, mass=m
                )
            lyap = system.compute_lyapunov_exponent(perturbation=1e-8, duration=50.0)
            if lyap > 0:
                lyapunov_times.append(1.0 / lyap)
        mean_lyap_time = np.mean(lyapunov_times) if lyapunov_times else 10.0

        seed_rollout = {}
        with torch.no_grad():
            for hf in horizon_factors:
                n_steps = int(hf * mean_lyap_time / config.save_interval)
                mses = []
                for traj in test_traj[:20]:
                    if traj.n_timesteps < k + n_steps:
                        continue
                    seed_frames = traj.density_fields[:k]
                    seed_norm = (seed_frames - train_ds.input_mean) / train_ds.input_std
                    seed_t = torch.from_numpy(seed_norm.astype(np.float32)).to(device)
                    pred_norm = fno_s.predict_trajectory(seed_t, n_steps)
                    pred = train_ds.denormalize_output(pred_norm).cpu().numpy()
                    true_idx = (k - 1) + n_steps
                    if n_steps < len(pred) and true_idx < traj.n_timesteps:
                        mse = float(np.mean((pred[n_steps] - traj.density_fields[true_idx]) ** 2))
                        mses.append(mse)
                seed_rollout[str(hf)] = float(np.mean(mses)) if mses else float('nan')

        rollout_mses_all.append(seed_rollout)

        seed_entry = {
            'seed': model_seed,
            'best_val_loss': best_val,
            'n_epochs_trained': len(history['train_loss']),
            'rollout_mses': seed_rollout,
        }
        results['seeds'].append(seed_entry)
        print(f"  Best val loss: {best_val:.6f}")
        for hf, mse in seed_rollout.items():
            print(f"  Rollout {hf}x Lyap: MSE = {mse:.6f}")

    # Summary statistics
    results['summary'] = {
        'val_loss_mean': float(np.mean(best_val_losses)),
        'val_loss_std': float(np.std(best_val_losses)),
        'val_loss_min': float(np.min(best_val_losses)),
        'val_loss_max': float(np.max(best_val_losses)),
    }

    # Aggregate rollout MSEs across seeds per horizon
    for hf_str in [str(hf) for hf in horizon_factors]:
        vals = [r[hf_str] for r in rollout_mses_all if not np.isnan(r.get(hf_str, float('nan')))]
        if vals:
            results['summary'][f'rollout_{hf_str}x_mean'] = float(np.mean(vals))
            results['summary'][f'rollout_{hf_str}x_std'] = float(np.std(vals))

    print("\n" + "=" * 60)
    print("MULTI-SEED SUMMARY")
    print("=" * 60)
    print(f"  Val loss: {results['summary']['val_loss_mean']:.6f} "
          f"± {results['summary']['val_loss_std']:.6f}")
    for hf_str in [str(hf) for hf in horizon_factors]:
        mean_key = f'rollout_{hf_str}x_mean'
        std_key = f'rollout_{hf_str}x_std'
        if mean_key in results['summary']:
            print(f"  Rollout {hf_str}x Lyap: "
                  f"{results['summary'][mean_key]:.6f} ± {results['summary'][std_key]:.6f}")

    return results


# =============================================================================
# EXPERIMENT 10: COMPUTATIONAL COST PROFILING
# =============================================================================

def experiment_10_computational_cost(
    config: ExperimentConfig,
    dataset: Dict,
) -> Dict:
    """
    Measure wall-clock timings for the physics simulator and the trained FNO.

    JCP reviewers expect a clear speedup table showing:
      1. Event-driven simulation time for one trajectory
      2. FNO single-step inference time
      3. FNO multi-step rollout time (100 and 500 steps)
      4. Speedup factor

    All timings are averaged over multiple repetitions for stability.
    """
    print("=" * 60)
    print("EXPERIMENT 10: Computational Cost Profiling")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")

    results = {
        'device': device,
        'simulator': {},
        'fno_inference': {},
        'speedup': {},
    }

    # ── 1. Physics simulator timing ──────────────────────────────────────
    print("\n1. Timing physics simulator...")

    sim_durations_per_step = []
    n_sim_trials = 5
    for trial in range(n_sim_trials):
        system = create_random_system(
            n_particles=8,
            track_length=config.track_length,
            seed=config.seed + trial
        )
        t_start = time.perf_counter()
        system.evolve(duration=config.simulation_time, save_interval=config.save_interval)
        t_end = time.perf_counter()

        n_steps_sim = int(config.simulation_time / config.save_interval)
        per_step = (t_end - t_start) / n_steps_sim
        sim_durations_per_step.append(per_step)

    results['simulator'] = {
        'n_trials': n_sim_trials,
        'simulation_time': config.simulation_time,
        'n_steps': int(config.simulation_time / config.save_interval),
        'total_time_mean_s': float(np.mean(sim_durations_per_step) * int(config.simulation_time / config.save_interval)),
        'per_step_mean_ms': float(np.mean(sim_durations_per_step) * 1000),
        'per_step_std_ms': float(np.std(sim_durations_per_step) * 1000),
    }
    print(f"  Simulator: {results['simulator']['per_step_mean_ms']:.3f} "
          f"± {results['simulator']['per_step_std_ms']:.3f} ms/step  "
          f"(total: {results['simulator']['total_time_mean_s']:.2f} s for "
          f"{results['simulator']['n_steps']} steps)")

    # ── 2. Load or build a trained FNO ───────────────────────────────────
    print("\n2. Loading trained FNO for inference timing...")

    # Try to find an existing checkpoint
    ckpt_path = os.path.join(config.output_dir, 'fno_checkpoint.pt')
    fno = FourierNeuralOperator(
        n_bins=config.n_density_bins,
        n_modes=config.fno_modes,
        hidden_channels=config.fno_hidden_channels,
        n_layers=config.fno_layers,
        delta_t=config.save_interval,
        k_history=config.k_history,
        lstm_hidden=config.lstm_hidden,
        lstm_layers=config.lstm_layers,
    ).to(device)

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        fno.load_state_dict(ckpt['model_state_dict'])
        print(f"  Loaded checkpoint from {ckpt_path}")
    else:
        print(f"  WARNING: No checkpoint at {ckpt_path}, using random weights (timings still valid)")

    fno.eval()

    # ── 3. FNO single-step inference timing ──────────────────────────────
    print("\n3. Timing FNO single-step inference...")
    k = config.k_history
    dummy_input = torch.randn(1, k, config.n_density_bins, device=device)

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            fno(dummy_input)
    if device == 'cuda':
        torch.cuda.synchronize()

    n_timing_reps = 100
    t_start = time.perf_counter()
    for _ in range(n_timing_reps):
        with torch.no_grad():
            fno(dummy_input)
    if device == 'cuda':
        torch.cuda.synchronize()
    t_end = time.perf_counter()

    single_step_ms = (t_end - t_start) / n_timing_reps * 1000
    results['fno_inference']['single_step_ms'] = float(single_step_ms)
    print(f"  FNO single-step: {single_step_ms:.3f} ms")

    # ── 4. FNO multi-step rollout timing ─────────────────────────────────
    print("\n4. Timing FNO multi-step rollout...")

    seed_window = torch.randn(k, config.n_density_bins, device=device)

    for n_rollout in [100, 500]:
        # Warmup
        with torch.no_grad():
            fno.predict_trajectory(seed_window, min(n_rollout, 10))
        if device == 'cuda':
            torch.cuda.synchronize()

        n_reps = 5
        t_start = time.perf_counter()
        for _ in range(n_reps):
            with torch.no_grad():
                fno.predict_trajectory(seed_window, n_rollout)
        if device == 'cuda':
            torch.cuda.synchronize()
        t_end = time.perf_counter()

        total_ms = (t_end - t_start) / n_reps * 1000
        per_step_ms = total_ms / n_rollout
        results['fno_inference'][f'rollout_{n_rollout}_total_ms'] = float(total_ms)
        results['fno_inference'][f'rollout_{n_rollout}_per_step_ms'] = float(per_step_ms)
        print(f"  FNO {n_rollout}-step rollout: {total_ms:.1f} ms total, "
              f"{per_step_ms:.3f} ms/step")

    # ── 5. Speedup calculation ───────────────────────────────────────────
    sim_per_step = results['simulator']['per_step_mean_ms']
    fno_per_step = results['fno_inference']['single_step_ms']
    speedup = sim_per_step / fno_per_step if fno_per_step > 0 else float('inf')

    if speedup >= 1.0:
        speedup_note = (f'FNO is {speedup:.1f}× faster per step than event-driven simulation')
    else:
        speedup_note = (
            f'Event-driven simulation is {1/speedup:.1f}× faster per step than FNO on {device}. '
            f'For this 1D system the analytical inter-collision dynamics are cheap. '
            f'The FNO advantage is density prediction beyond the Lyapunov horizon, '
            f'not raw simulation speed.'
        )

    results['speedup'] = {
        'single_step': float(speedup),
        'note': speedup_note,
    }

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("COMPUTATIONAL COST SUMMARY")
    print("=" * 60)
    print(f"  {'Component':<35} {'Time':>10}")
    print("  " + "-" * 47)
    print(f"  {'Simulator (per step)':<35} {sim_per_step:>8.3f} ms")
    print(f"  {'FNO single-step inference':<35} {fno_per_step:>8.3f} ms")
    if 'rollout_100_per_step_ms' in results['fno_inference']:
        print(f"  {'FNO 100-step rollout (per step)':<35} "
              f"{results['fno_inference']['rollout_100_per_step_ms']:>8.3f} ms")
    if 'rollout_500_per_step_ms' in results['fno_inference']:
        print(f"  {'FNO 500-step rollout (per step)':<35} "
              f"{results['fno_inference']['rollout_500_per_step_ms']:>8.3f} ms")
    print(f"\n  Speedup: {speedup:.1f}× (single-step)")

    return results


# =============================================================================
# RUN MODES — trajectory / simulation_time budgets per scale
# =============================================================================

# Each entry: (n_training, n_validation, n_test, simulation_time)
RUN_MODE_PRESETS: Dict[str, Tuple[int, int, int, float]] = {
    #              train   val   test   sim_time
    "short-test":  (   10,   5,    5,   20.0),   # ~1–2 min  — pure smoke test
    "quick":       (   50,  10,   10,   50.0),   # ~10–20 min
    "intermediate":(  200,  40,   40,  200.0),   # ~1–2 h
    "full":        ( 1000, 100,  100,  500.0),   # paper-spec  (~8–24 h)
    "extended":    ( 3000, 300,  300,  500.0),   # statistical robustness
}


# =============================================================================
# TIMING HELPERS
# =============================================================================

class RunTimer:
    """
    Lightweight experiment-level timer that logs each phase and prints
    a running ETA based on elapsed wall-clock time.
    """

    def __init__(self, total_experiments: int):
        self.total   = total_experiments
        self.done    = 0
        self.records: List[Dict] = []
        self._wall_start = time.perf_counter()

    # ------------------------------------------------------------------
    def start(self, name: str) -> float:
        """Mark the start of an experiment phase; returns a start timestamp."""
        t = time.perf_counter()
        print(f"\n[TIMER] {name} — starting at +{t - self._wall_start:.1f}s")
        return t

    def stop(self, name: str, t_start: float) -> float:
        """Mark the end of a phase. Logs elapsed and prints running ETA."""
        elapsed = time.perf_counter() - t_start
        self.done += 1
        self.records.append({"name": name, "elapsed_s": elapsed})

        total_elapsed = time.perf_counter() - self._wall_start
        avg_per_exp   = total_elapsed / self.done
        remaining     = (self.total - self.done) * avg_per_exp

        print(f"[TIMER] {name} — done in {_fmt(elapsed)}  "
              f"| total so far: {_fmt(total_elapsed)}  "
              f"| ETA remaining: ~{_fmt(remaining)}  "
              f"({self.done}/{self.total} phases)")
        return elapsed

    def summary(self) -> str:
        """Return a formatted timing summary table."""
        lines = [
            "",
            "=" * 60,
            "TIMING SUMMARY",
            "=" * 60,
            f"  {'Experiment':<35} {'Elapsed':>10}",
            "  " + "-" * 47,
        ]
        total = 0.0
        for rec in self.records:
            lines.append(f"  {rec['name']:<35} {_fmt(rec['elapsed_s']):>10}")
            total += rec["elapsed_s"]
        wall = time.perf_counter() - self._wall_start
        lines += [
            "  " + "-" * 47,
            f"  {'Total wall-clock':<35} {_fmt(wall):>10}",
            "=" * 60,
        ]
        return "\n".join(lines)


def _fmt(seconds: float) -> str:
    """Human-readable duration: e.g. 3h 02m 15s."""
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {sec:02d}s"
    if m:
        return f"{m}m {sec:02d}s"
    return f"{sec}s"


# =============================================================================
# MAIN ORCHESTRATION
# =============================================================================

def run_all_experiments(
    config: Optional[ExperimentConfig] = None,
    smoke_test: bool = False,
) -> Tuple[Dict, str]:
    """
    Orchestrate all Paper 1 experiments and save results to disk.

    Parameters
    ----------
    config     : Pre-built ExperimentConfig; if None one is constructed.
    smoke_test : If True, forward smoke-test flag to Experiment 7b.

    Returns
    -------
    (all_results, run_dir)
    """
    if config is None:
        config = ExperimentConfig(
            experiment_name="paper1_intermediate_run",
            seed=42,
            n_training_trajectories=200,
            n_validation_trajectories=40,
            n_test_trajectories=40,
            simulation_time=200.0,
        )

    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(config.output_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    config.save(os.path.join(run_dir, "config.json"))

    print("\n" + "=" * 70)
    print(f"PAPER 1 EXPERIMENTS  —  {config.experiment_name}")
    print(f"Run dir : {run_dir}")
    print(f"Device  : {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Dataset : train={config.n_training_trajectories}  "
          f"val={config.n_validation_trajectories}  "
          f"test={config.n_test_trajectories}  "
          f"sim_time={config.simulation_time}s")
    print("=" * 70)

    # 13 phases: exps 1–7 + 3b + 7b + 8 + 9 + 10 + save
    timer = RunTimer(total_experiments=13)
    all_results: Dict = {}

    # ------------------------------------------------------------------
    # Experiment 1: Dataset generation
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RUNNING EXPERIMENT 1: Dataset Generation")
    print("=" * 70)
    t0 = timer.start("Exp 1: Dataset Generation")
    dataset = experiment_1_generate_dataset(config)
    all_results["experiment_1_dataset"] = dataset["statistics"]
    timer.stop("Exp 1: Dataset Generation", t0)

    with open(os.path.join(run_dir, "dataset.json"), "w") as f:
        json.dump(dataset, f, default=_json_safe)
    print(f"  Dataset saved to {run_dir}/dataset.json")

    # ------------------------------------------------------------------
    # Experiment 2: Lyapunov characterisation
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RUNNING EXPERIMENT 2: Lyapunov Characterisation")
    print("=" * 70)
    t0 = timer.start("Exp 2: Lyapunov Characterisation")
    lyapunov_results = experiment_2_lyapunov_characterization(config)
    all_results["experiment_2_lyapunov"] = lyapunov_results
    timer.stop("Exp 2: Lyapunov Characterisation", t0)

    # ------------------------------------------------------------------
    # Experiment 3: FNO training
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RUNNING EXPERIMENT 3: FNO Training")
    print("=" * 70)
    t0 = timer.start("Exp 3: FNO Training")
    training_results, fno, train_dataset = experiment_3_fno_training(config, dataset)
    all_results["experiment_3_training"] = training_results
    timer.stop("Exp 3: FNO Training", t0)

    model_path = os.path.join(run_dir, "fno_checkpoint.pt")
    save_model(fno, model_path, metadata={"config": asdict(config)})
    print(f"  FNO checkpoint saved to {model_path}")

    # ------------------------------------------------------------------
    # Experiment 3b: Curriculum fine-tuning
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RUNNING EXPERIMENT 3b: Curriculum Fine-Tuning (Scheduled Sampling)")
    print("=" * 70)
    t0 = timer.start("Exp 3b: Curriculum Fine-Tuning")
    finetuning_results, fno_finetuned = experiment_3b_fno_curriculum_finetuning(
        config, fno, dataset, train_dataset
    )
    all_results["experiment_3b_curriculum"] = finetuning_results
    timer.stop("Exp 3b: Curriculum Fine-Tuning", t0)

    finetuned_path = os.path.join(run_dir, "fno_finetuned_checkpoint.pt")
    save_model(fno_finetuned, finetuned_path, metadata={"config": asdict(config)})
    print(f"  Fine-tuned FNO checkpoint saved to {finetuned_path}")

    # ------------------------------------------------------------------
    # Experiment 4: Lyapunov bypass  (uses curriculum-fine-tuned model)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RUNNING EXPERIMENT 4: Lyapunov Horizon Bypass")
    print("=" * 70)
    t0 = timer.start("Exp 4: Lyapunov Bypass")
    bypass_results = experiment_4_lyapunov_bypass(config, fno_finetuned, dataset, train_dataset)
    all_results["experiment_4_bypass"] = bypass_results
    timer.stop("Exp 4: Lyapunov Bypass", t0)

    # ------------------------------------------------------------------
    # Experiment 5: Ablations
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RUNNING EXPERIMENT 5: Ablation Studies")
    print("=" * 70)
    t0 = timer.start("Exp 5: Ablation Studies")
    ablation_results = experiment_5_ablations(config, dataset, train_dataset)
    all_results["experiment_5_ablations"] = ablation_results
    timer.stop("Exp 5: Ablation Studies", t0)

    # ------------------------------------------------------------------
    # Experiment 6: Baselines
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RUNNING EXPERIMENT 6: Baseline Comparisons")
    print("=" * 70)
    t0 = timer.start("Exp 6: Baseline Comparisons")
    baseline_results = experiment_6_baselines(config, dataset)
    all_results["experiment_6_baselines"] = baseline_results
    timer.stop("Exp 6: Baseline Comparisons", t0)

    # ------------------------------------------------------------------
    # Experiment 7: Inverse problem (collision-graph, Framing B)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RUNNING EXPERIMENT 7: Inverse Problem (Framing B — collision encoder)")
    print("=" * 70)
    t0 = timer.start("Exp 7: Inverse Problem (Framing B)")
    inverse_results = experiment_7_inverse_problem(config, dataset)
    all_results["experiment_7_inverse"] = inverse_results
    timer.stop("Exp 7: Inverse Problem (Framing B)", t0)

    # ------------------------------------------------------------------
    # Experiment 7b: FNO-in-the-loop inverse problem (Framing A)
    # Requires fno_checkpoint.pt written by Exp 3 to be found in output_dir.
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RUNNING EXPERIMENT 7b: FNO-In-The-Loop Inverse Problem (Framing A)")
    print("=" * 70)
    t0 = timer.start("Exp 7b: FNO Inverse Problem (Framing A)")
    try:
        # The checkpoint is saved to run_dir; patch output_dir so that
        # experiment_7b can find it via cfg.output_dir / cfg.checkpoint_name.
        exp7b_config = ExperimentConfig(
            **{**asdict(config),
               "output_dir": run_dir,
               "experiment_name": config.experiment_name + "_7b"})
        from experiment_7b_inverse_fno import experiment_7b_inverse_fno as _exp7b
        inverse_fno_results = _exp7b(
            config=exp7b_config,
            dataset=dataset,
            smoke_test=smoke_test,
        )
        all_results["experiment_7b_inverse_fno"] = inverse_fno_results
    except Exception as exc:
        print(f"  WARNING: Experiment 7b failed with: {exc}")
        print("  (Ensure experiment_7b_inverse_fno.py is on the Python path.)")
        all_results["experiment_7b_inverse_fno"] = {"error": str(exc)}
    timer.stop("Exp 7b: FNO Inverse Problem (Framing A)", t0)

    # ------------------------------------------------------------------
    # Experiment 8: Grid convergence study
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RUNNING EXPERIMENT 8: Grid Convergence Study")
    print("=" * 70)
    t0 = timer.start("Exp 8: Grid Convergence")
    grid_results = experiment_8_grid_convergence(config, dataset)
    all_results["experiment_8_grid_convergence"] = grid_results
    timer.stop("Exp 8: Grid Convergence", t0)

    # ------------------------------------------------------------------
    # Experiment 9: Multi-seed training
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RUNNING EXPERIMENT 9: Multi-Seed Training")
    print("=" * 70)
    t0 = timer.start("Exp 9: Multi-Seed Training")
    seed_results = experiment_9_multi_seed(config, dataset, n_seeds=3)
    all_results["experiment_9_multi_seed"] = seed_results
    timer.stop("Exp 9: Multi-Seed Training", t0)

    # ------------------------------------------------------------------
    # Experiment 10: Computational cost profiling
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RUNNING EXPERIMENT 10: Computational Cost Profiling")
    print("=" * 70)
    t0 = timer.start("Exp 10: Computational Cost")
    cost_results = experiment_10_computational_cost(config, dataset)
    all_results["experiment_10_computational_cost"] = cost_results
    timer.stop("Exp 10: Computational Cost", t0)

    # ------------------------------------------------------------------
    # Save everything
    # ------------------------------------------------------------------
    t0 = timer.start("Saving results")
    results_path = os.path.join(run_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=_json_safe)
    print(f"  Results saved to {results_path}")
    timer.stop("Saving results", t0)

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print(timer.summary())
    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE")
    print(f"Results saved to: {run_dir}")
    print("=" * 70)

    return all_results, run_dir


def _json_safe(obj):
    """JSON serialiser for numpy scalars produced by experiments."""
    import numpy as np  # local just in case
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")


# =============================================================================
# COMMAND-LINE ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Paper 1 experiments: Neural Operators for Density Evolution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Run modes (--mode):
  short-test    10 train /  5 val /  5 test / 20 s   — ~1-2 min  (smoke test)
  quick         50 train / 10 val / 10 test / 50 s   — ~10-20 min
  intermediate 200 train / 40 val / 40 test / 200 s  — ~1-2 h
  full        1000 train /100 val /100 test / 500 s  — ~8-24 h   (paper spec)
  extended    3000 train /300 val /300 test / 500 s  — for statistical robustness
""",
    )
    parser.add_argument(
        "--mode",
        choices=list(RUN_MODE_PRESETS.keys()),
        default="intermediate",
        help="Scale of the experiment run (default: intermediate)",
    )
    parser.add_argument("--output-dir", default="./results/paper1",
                        help="Root directory for all outputs")
    parser.add_argument("--seed", type=int, default=42,
                        help="Global random seed")
    parser.add_argument("--name", default=None,
                        help="Override experiment name (default: paper1_<mode>)")
    parser.add_argument("--smoke-test-7b", action="store_true",
                        help="Run Experiment 7b in smoke-test mode (saves time)")
    args = parser.parse_args()

    n_train, n_val, n_test, sim_time = RUN_MODE_PRESETS[args.mode]
    exp_name = args.name or f"paper1_{args.mode}"

    print(f"\nRun mode : {args.mode}")
    print(f"  train={n_train}  val={n_val}  test={n_test}  sim_time={sim_time}s")
    print(f"  output_dir={args.output_dir}  seed={args.seed}")

    config = ExperimentConfig(
        experiment_name=exp_name,
        seed=args.seed,
        output_dir=args.output_dir,
        n_training_trajectories=n_train,
        n_validation_trajectories=n_val,
        n_test_trajectories=n_test,
        simulation_time=sim_time,
        # All FNO / LSTM architecture params stay at their ExperimentConfig defaults
    )

    results, output_dir = run_all_experiments(
        config,
        smoke_test=args.smoke_test_7b,
    )
    print(f"\nAll done. Results in: {output_dir}")

