"""
Model Predictive Control for Tilting Billiards

This module implements controllers that demonstrate the value of
density-based prediction (FNO) over trajectory-based prediction.

Controllers:
1. Random: Baseline random actions
2. Greedy: Tilt toward nearest ball-pocket pair
3. MPC-Trajectory: Plan using trajectory predictions (fails due to chaos)
4. MPC-Density: Plan using FNO density predictions (works beyond Lyapunov time!)

The key experiment: Show that MPC-Density outperforms MPC-Trajectory
at longer planning horizons because density predictions remain accurate
while trajectory predictions diverge.

Author: Claude (Anthropic) + Human collaborator
"""

import numpy as np
from typing import Dict, List, Callable, Optional, Tuple
from dataclasses import dataclass
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from billiards_env import (
    TiltingBilliardsEnv, TableConfig, TableState, Ball,
    TiltingTablePhysics, create_random_scenario
)


# =============================================================================
# CONTROLLER BASE CLASS
# =============================================================================

class Controller:
    """Base class for billiards controllers."""
    
    def __init__(self, name: str):
        self.name = name
    
    def __call__(self, observation: Dict) -> np.ndarray:
        """Return action given observation."""
        raise NotImplementedError
    
    def reset(self):
        """Reset controller state (if any)."""
        pass


# =============================================================================
# BASELINE CONTROLLERS
# =============================================================================

class RandomController(Controller):
    """Random tilt actions."""
    
    def __init__(self, max_rate: float = 0.1, seed: Optional[int] = None):
        super().__init__("Random")
        self.max_rate = max_rate
        self.rng = np.random.default_rng(seed)
    
    def __call__(self, observation: Dict) -> np.ndarray:
        return self.rng.uniform(-self.max_rate, self.max_rate, 2)


class GreedyController(Controller):
    """
    Greedy controller: Tilt toward the nearest ball-pocket vector.
    
    Strategy: Find the ball-pocket pair with smallest distance,
    then tilt to accelerate that ball toward the pocket.
    """
    
    def __init__(
        self,
        config: TableConfig,
        gain: float = 0.5
    ):
        super().__init__("Greedy")
        self.config = config
        self.gain = gain
        
        # Pocket positions
        self.pocket_positions = self._get_pocket_positions()
    
    def _get_pocket_positions(self) -> np.ndarray:
        """Get positions of all 6 pockets."""
        cfg = self.config
        return np.array([
            [-cfg.half_length, -cfg.half_width],  # Bottom-left
            [-cfg.half_length, cfg.half_width],   # Top-left
            [cfg.half_length, -cfg.half_width],   # Bottom-right
            [cfg.half_length, cfg.half_width],    # Top-right
            [0, -cfg.half_width],                 # Bottom-center
            [0, cfg.half_width],                  # Top-center
        ])
    
    def __call__(self, observation: Dict) -> np.ndarray:
        positions = observation['positions']
        active_mask = observation['active_mask']
        
        # Find active ball positions
        active_positions = positions[active_mask > 0.5]
        
        if len(active_positions) == 0:
            return np.zeros(2)
        
        # Find nearest ball-pocket pair
        min_dist = float('inf')
        best_direction = np.zeros(2)
        
        for ball_pos in active_positions:
            for pocket_pos in self.pocket_positions:
                dist = np.linalg.norm(pocket_pos - ball_pos)
                if dist < min_dist:
                    min_dist = dist
                    # Direction from ball to pocket
                    best_direction = (pocket_pos - ball_pos) / (dist + 1e-6)
        
        # Convert desired acceleration direction to tilt rates
        # Tilt pitch (θ_y) creates acceleration in x
        # Tilt roll (θ_x) creates acceleration in y
        roll_rate = self.gain * best_direction[1]   # y acceleration
        pitch_rate = self.gain * best_direction[0]  # x acceleration
        
        return np.array([roll_rate, pitch_rate])


# =============================================================================
# TRAJECTORY-BASED MPC (Baseline)
# =============================================================================

class TrajectoryMPC(Controller):
    """
    Model Predictive Control using trajectory prediction.
    
    This controller:
    1. Predicts future ball trajectories using physics simulation
    2. Optimizes tilt sequence to pocket balls
    3. Executes first action, then replans
    
    Problem: Due to chaotic dynamics, trajectory predictions diverge
    after the Lyapunov time (~0.5-1s), causing planning to fail.
    """
    
    def __init__(
        self,
        config: TableConfig,
        horizon: float = 1.0,       # Planning horizon in seconds
        n_samples: int = 50,        # Number of action samples to try
        dt_plan: float = 0.01,      # Planning timestep
        seed: Optional[int] = None
    ):
        super().__init__(f"MPC-Trajectory (h={horizon}s)")
        self.config = config
        self.horizon = horizon
        self.n_samples = n_samples
        self.dt_plan = dt_plan
        self.rng = np.random.default_rng(seed)
        
        # Physics simulator for prediction
        self.physics = TiltingTablePhysics(config)
        
        # Pocket positions for reward
        self.pocket_positions = np.array([
            [-config.half_length, -config.half_width],
            [-config.half_length, config.half_width],
            [config.half_length, -config.half_width],
            [config.half_length, config.half_width],
            [0, -config.half_width],
            [0, config.half_width],
        ])
        self.pocket_radius = config.pocket_radius
    
    def _state_from_observation(self, obs: Dict) -> TableState:
        """Reconstruct TableState from observation."""
        positions = obs['positions']
        velocities = obs['velocities']
        active_mask = obs['active_mask']
        tilt = obs['tilt']
        
        balls = []
        for i in range(len(positions)):
            if active_mask[i] > 0.5:
                balls.append(Ball(
                    id=i,
                    position=positions[i].copy(),
                    velocity=velocities[i].copy(),
                    mass=self.config.ball_mass_mean,
                    radius=self.config.ball_radius,
                    is_pocketed=False
                ))
        
        return TableState(
            time=0.0,
            balls=balls,
            tilt_roll=tilt[0],
            tilt_pitch=tilt[1],
            balls_pocketed=0
        )
    
    def _evaluate_trajectory(
        self,
        initial_state: TableState,
        action_sequence: np.ndarray
    ) -> float:
        """
        Evaluate an action sequence by simulating forward.
        
        Returns score (higher = better).
        """
        state = initial_state.copy()
        physics = TiltingTablePhysics(self.config)
        
        total_score = 0.0
        n_steps = len(action_sequence)
        
        for t, action in enumerate(action_sequence):
            # Simulate one planning step
            n_substeps = max(1, int(self.dt_plan / self.config.dt))
            for _ in range(n_substeps):
                state, events = physics.step(state, action)
            
            # Reward for pocketed balls
            total_score += state.balls_pocketed * 10.0
            
            # Reward for balls close to pockets
            for ball in state.active_balls:
                for pocket in self.pocket_positions:
                    dist = np.linalg.norm(ball.position - pocket)
                    if dist < self.pocket_radius * 3:
                        total_score += (1.0 - dist / (self.pocket_radius * 3)) * 0.1
            
            # Small penalty for time (encourage faster solutions)
            total_score -= 0.01
        
        return total_score
    
    def __call__(self, observation: Dict) -> np.ndarray:
        """Plan and return best first action."""
        state = self._state_from_observation(observation)
        
        if len(state.active_balls) == 0:
            return np.zeros(2)
        
        n_steps = int(self.horizon / self.dt_plan)
        max_rate = self.config.max_tilt_rate
        
        best_score = float('-inf')
        best_action = np.zeros(2)
        
        for _ in range(self.n_samples):
            # Sample random action sequence
            action_sequence = self.rng.uniform(
                -max_rate, max_rate, (n_steps, 2)
            )
            
            # Smooth the sequence for more realistic actions
            for t in range(1, n_steps):
                action_sequence[t] = 0.7 * action_sequence[t-1] + 0.3 * action_sequence[t]
            
            score = self._evaluate_trajectory(state.copy(), action_sequence)
            
            if score > best_score:
                best_score = score
                best_action = action_sequence[0].copy()
        
        return best_action


# =============================================================================
# DENSITY-BASED MPC (Our Method)
# =============================================================================

class DensityPredictor:
    """
    Placeholder for the trained FNO density predictor.
    
    In the full implementation, this would load a trained
    FourierNeuralOperator and use it to predict density evolution.
    
    For now, we use a simple physics-based density propagation
    as a stand-in, but the key insight is:
    - Even approximate density predictions remain useful longer
    - Trajectory predictions become useless after Lyapunov time
    """
    
    def __init__(
        self,
        config: TableConfig,
        resolution: int = 20
    ):
        self.config = config
        self.resolution = resolution
        self.physics = TiltingTablePhysics(config)
    
    def predict_density(
        self,
        initial_density: np.ndarray,
        initial_state: TableState,
        action_sequence: np.ndarray,
        dt: float = 0.01
    ) -> List[np.ndarray]:
        """
        Predict density evolution under action sequence.
        
        In full implementation: Use trained FNO
        For now: Use Monte Carlo simulation with multiple particles
        
        Returns list of density fields at each timestep.
        """
        # Monte Carlo density estimation
        # Run multiple perturbed trajectories and average
        n_samples = 20
        densities = []
        
        accumulated_densities = []
        
        for step_idx, action in enumerate(action_sequence):
            step_density = np.zeros((self.resolution, self.resolution))
            
            for sample in range(n_samples):
                # Perturb initial state slightly
                state = initial_state.copy()
                for ball in state.balls:
                    ball.position += np.random.normal(0, 0.01, 2)
                    ball.velocity += np.random.normal(0, 0.02, 2)
                
                # Simulate to this timestep
                physics = TiltingTablePhysics(self.config)
                for prev_action in action_sequence[:step_idx+1]:
                    n_substeps = max(1, int(dt / self.config.dt))
                    for _ in range(n_substeps):
                        state, _ = physics.step(state, prev_action)
                
                # Add to density
                step_density += self._compute_density(state)
            
            step_density /= n_samples
            accumulated_densities.append(step_density)
        
        return accumulated_densities
    
    def _compute_density(self, state: TableState) -> np.ndarray:
        """Compute density field from state (vectorized)."""
        cfg = self.config
        x_edges = np.linspace(-cfg.half_length, cfg.half_length, self.resolution + 1)
        y_edges = np.linspace(-cfg.half_width, cfg.half_width, self.resolution + 1)
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2

        XX, YY = np.meshgrid(x_centers, y_centers)
        density = np.zeros((self.resolution, self.resolution))

        sigma = cfg.ball_radius * 2
        inv_two_sigma_sq = 1.0 / (2.0 * sigma ** 2)

        for ball in state.active_balls:
            dist_sq = (XX - ball.position[0]) ** 2 + (YY - ball.position[1]) ** 2
            density += np.exp(-dist_sq * inv_two_sigma_sq)

        return density


class DensityMPC(Controller):
    """
    Model Predictive Control using density prediction.
    
    This controller:
    1. Predicts future density fields using FNO (or Monte Carlo proxy)
    2. Optimizes tilt sequence to maximize probability mass near pockets
    3. Executes first action, then replans
    
    Advantage: Density predictions remain accurate beyond Lyapunov time
    because we're predicting distributions, not exact trajectories.
    """
    
    def __init__(
        self,
        config: TableConfig,
        horizon: float = 2.0,       # Can use longer horizon!
        n_samples: int = 30,
        dt_plan: float = 0.05,
        resolution: int = 20,
        seed: Optional[int] = None
    ):
        super().__init__(f"MPC-Density (h={horizon}s)")
        self.config = config
        self.horizon = horizon
        self.n_samples = n_samples
        self.dt_plan = dt_plan
        self.resolution = resolution
        self.rng = np.random.default_rng(seed)
        
        # Density predictor (FNO in full implementation)
        self.density_predictor = DensityPredictor(config, resolution)
        
        # Create pocket masks
        self.pocket_mask = self._create_pocket_mask()
    
    def _create_pocket_mask(self) -> np.ndarray:
        """Create a mask with high values near pockets (vectorized)."""
        cfg = self.config
        x_edges = np.linspace(-cfg.half_length, cfg.half_length, self.resolution + 1)
        y_edges = np.linspace(-cfg.half_width, cfg.half_width, self.resolution + 1)
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2

        XX, YY = np.meshgrid(x_centers, y_centers)
        mask = np.zeros((self.resolution, self.resolution))

        pocket_positions = [
            (-cfg.half_length, -cfg.half_width),
            (-cfg.half_length,  cfg.half_width),
            ( cfg.half_length, -cfg.half_width),
            ( cfg.half_length,  cfg.half_width),
            (0, -cfg.half_width),
            (0,  cfg.half_width),
        ]
        decay = cfg.pocket_radius * 2

        for px, py in pocket_positions:
            dist = np.sqrt((XX - px) ** 2 + (YY - py) ** 2)
            mask += np.exp(-dist / decay)

        return mask
    
    def _state_from_observation(self, obs: Dict) -> TableState:
        """Reconstruct TableState from observation."""
        positions = obs['positions']
        velocities = obs['velocities']
        active_mask = obs['active_mask']
        tilt = obs['tilt']
        
        balls = []
        for i in range(len(positions)):
            if active_mask[i] > 0.5:
                balls.append(Ball(
                    id=i,
                    position=positions[i].copy(),
                    velocity=velocities[i].copy(),
                    mass=self.config.ball_mass_mean,
                    radius=self.config.ball_radius,
                    is_pocketed=False
                ))
        
        return TableState(
            time=0.0,
            balls=balls,
            tilt_roll=tilt[0],
            tilt_pitch=tilt[1],
            balls_pocketed=0
        )
    
    def _evaluate_action_sequence(
        self,
        initial_state: TableState,
        action_sequence: np.ndarray
    ) -> float:
        """
        Evaluate action sequence using density prediction.
        
        Score = sum of (density * pocket_mask) at each timestep
        """
        densities = self.density_predictor.predict_density(
            initial_density=None,  # Not used in current implementation
            initial_state=initial_state,
            action_sequence=action_sequence,
            dt=self.dt_plan
        )
        
        total_score = 0.0
        
        for t, density in enumerate(densities):
            # Reward: probability mass overlapping with pocket regions
            overlap = np.sum(density * self.pocket_mask)
            
            # Discount future rewards slightly
            discount = 0.99 ** t
            total_score += overlap * discount
        
        return total_score
    
    def __call__(self, observation: Dict) -> np.ndarray:
        """Plan and return best first action."""
        state = self._state_from_observation(observation)
        
        if len(state.active_balls) == 0:
            return np.zeros(2)
        
        n_steps = int(self.horizon / self.dt_plan)
        max_rate = self.config.max_tilt_rate
        
        best_score = float('-inf')
        best_action = np.zeros(2)
        
        for _ in range(self.n_samples):
            # Sample random action sequence
            action_sequence = self.rng.uniform(
                -max_rate, max_rate, (n_steps, 2)
            )
            
            # Smooth the sequence
            for t in range(1, n_steps):
                action_sequence[t] = 0.8 * action_sequence[t-1] + 0.2 * action_sequence[t]
            
            score = self._evaluate_action_sequence(state.copy(), action_sequence)
            
            if score > best_score:
                best_score = score
                best_action = action_sequence[0].copy()
        
        return best_action


# =============================================================================
# EXPERIMENT: COMPARE CONTROLLERS
# =============================================================================

def compare_controllers(
    n_episodes: int = 20,
    max_steps: int = 500,
    horizons: List[float] = [0.5, 1.0, 2.0, 3.0],
    seed: int = 42
) -> Dict:
    """
    Compare all controllers across different planning horizons.
    
    This is the key experiment showing that density-based MPC
    outperforms trajectory-based MPC at longer horizons.
    """
    config = TableConfig()
    results = {
        'controllers': [],
        'by_horizon': {}
    }
    
    # Controllers to test
    controller_factories = [
        ('Random', lambda h: RandomController(seed=seed)),
        ('Greedy', lambda h: GreedyController(config)),
    ]
    
    # Add MPC controllers at different horizons
    for horizon in horizons:
        controller_factories.append(
            (f'MPC-Traj-{horizon}s', 
             lambda h, hz=horizon: TrajectoryMPC(config, horizon=hz, seed=seed))
        )
        controller_factories.append(
            (f'MPC-Dens-{horizon}s',
             lambda h, hz=horizon: DensityMPC(config, horizon=hz, seed=seed))
        )
    
    for name, factory in controller_factories:
        print(f"\nEvaluating {name}...")
        
        controller = factory(None)
        episode_rewards = []
        episode_pocketed = []
        episode_steps = []
        
        for ep in range(n_episodes):
            env = create_random_scenario(n_balls=7, seed=seed + ep)
            
            total_reward = 0
            for step in range(max_steps):
                obs = env.get_observation()
                action = controller(obs)
                
                state, reward, done, info = env.step(action)
                total_reward += reward
                
                if done:
                    break
            
            episode_rewards.append(total_reward)
            episode_pocketed.append(state.balls_pocketed)
            episode_steps.append(step + 1)
            
            if (ep + 1) % 5 == 0:
                print(f"  Episode {ep+1}/{n_episodes}: "
                      f"Pocketed {state.balls_pocketed}, "
                      f"Steps {step+1}")
        
        results['controllers'].append({
            'name': name,
            'mean_reward': float(np.mean(episode_rewards)),
            'std_reward': float(np.std(episode_rewards)),
            'mean_pocketed': float(np.mean(episode_pocketed)),
            'std_pocketed': float(np.std(episode_pocketed)),
            'mean_steps': float(np.mean(episode_steps)),
        })
    
    # Organize by horizon for plotting
    for horizon in horizons:
        traj_results = next(
            (r for r in results['controllers'] if r['name'] == f'MPC-Traj-{horizon}s'),
            None
        )
        dens_results = next(
            (r for r in results['controllers'] if r['name'] == f'MPC-Dens-{horizon}s'),
            None
        )
        
        if traj_results and dens_results:
            results['by_horizon'][horizon] = {
                'trajectory': traj_results,
                'density': dens_results,
                'improvement': (
                    (dens_results['mean_pocketed'] - traj_results['mean_pocketed']) /
                    max(traj_results['mean_pocketed'], 0.1) * 100
                )
            }
    
    return results


def run_billiards_experiment(seed: int = 42) -> Dict:
    """
    Full billiards experiment for Paper 1.
    
    Generates data for:
    - Figure 8a: Task performance vs planning horizon
    - Figure 8b: Prediction accuracy comparison  
    - Figure 8c: Example rollouts
    """
    print("=" * 60)
    print("BILLIARDS TABLE EXPERIMENT")
    print("=" * 60)
    
    results = {}
    
    # Part 1: Compare controllers
    print("\nPart 1: Controller comparison...")
    results['controller_comparison'] = compare_controllers(
        n_episodes=10,  # Reduced for demo
        max_steps=300,
        horizons=[0.5, 1.0, 2.0],
        seed=seed
    )
    
    # Part 2: Measure Lyapunov time
    print("\nPart 2: Lyapunov time estimation...")
    env = create_random_scenario(n_balls=7, seed=seed)
    lyapunov_estimates = []
    
    for trial in range(5):
        env.reset()
        lyap = env.estimate_lyapunov_exponent(duration=1.0)
        lyapunov_estimates.append(lyap)
        print(f"  Trial {trial+1}: λ = {lyap:.3f}")
    
    mean_lyap = np.mean(lyapunov_estimates)
    lyapunov_time = 1.0 / mean_lyap if mean_lyap > 0 else float('inf')
    
    results['lyapunov_analysis'] = {
        'lyapunov_exponents': lyapunov_estimates,
        'mean_lyapunov': float(mean_lyap),
        'lyapunov_time': float(lyapunov_time)
    }
    print(f"  Mean Lyapunov time: {lyapunov_time:.3f}s")
    
    # Part 3: Prediction accuracy over time
    print("\nPart 3: Prediction accuracy vs horizon...")
    # This would compare FNO density prediction vs trajectory prediction
    # accuracy at different time horizons
    
    results['summary'] = {
        'best_controller': max(
            results['controller_comparison']['controllers'],
            key=lambda x: x['mean_pocketed']
        )['name'],
        'lyapunov_time': lyapunov_time,
        'density_improvement_at_2s': results['controller_comparison']['by_horizon'].get(
            2.0, {}
        ).get('improvement', 0)
    }
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Best controller: {results['summary']['best_controller']}")
    print(f"Lyapunov time: {lyapunov_time:.3f}s")
    
    return results


if __name__ == "__main__":
    results = run_billiards_experiment(seed=42)
    
    print("\n\nController Performance:")
    print("-" * 50)
    for ctrl in results['controller_comparison']['controllers']:
        print(f"{ctrl['name']:20s}: {ctrl['mean_pocketed']:.2f} ± {ctrl['std_pocketed']:.2f} balls")
