"""
Tilting Billiards Table Environment

A robotics-relevant testbed for neural operator-based control.

The agent controls roll and pitch of a table to guide balls into pockets.
This demonstrates the value of density prediction for chaotic collision dynamics.

Physical setup:
- Rectangular table that pivots around its center
- Multiple balls with slight mass variation (creates chaos)
- 6 pockets (4 corners + 2 sides)
- Gravity acts through tilt, causing balls to roll

Control challenge:
- Ball-ball collisions are chaotic
- Long-horizon planning requires predicting density, not trajectories
- FNO enables planning beyond the Lyapunov time

Author: Claude (Anthropic) + Human collaborator
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from enum import Enum


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class TableConfig:
    """Physical parameters of the billiards table."""
    
    # Table dimensions (meters)
    length: float = 1.0
    width: float = 0.5
    
    # Ball properties
    ball_radius: float = 0.0285  # 5.7 cm diameter
    ball_mass_mean: float = 0.165  # 165 grams
    ball_mass_std: float = 0.005  # Slight variation for chaos
    
    # Pocket properties
    pocket_radius: float = 0.055  # 11 cm diameter opening
    
    # Physics
    gravity: float = 9.81
    rolling_friction: float = 0.01  # Rolling resistance coefficient
    collision_restitution: float = 0.95  # Coefficient of restitution
    wall_restitution: float = 0.8
    
    # Control limits
    max_tilt_angle: float = np.radians(5)  # ±5 degrees
    max_tilt_rate: float = np.radians(10)  # 10 deg/s max angular velocity
    
    # Simulation
    dt: float = 0.001  # 1ms timestep for accurate collision detection
    
    @property
    def half_length(self) -> float:
        return self.length / 2
    
    @property
    def half_width(self) -> float:
        return self.width / 2


@dataclass
class Ball:
    """A single ball on the table."""
    id: int
    position: np.ndarray  # [x, y] in table frame
    velocity: np.ndarray  # [vx, vy]
    mass: float
    radius: float
    is_pocketed: bool = False
    
    def copy(self) -> 'Ball':
        return Ball(
            id=self.id,
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            mass=self.mass,
            radius=self.radius,
            is_pocketed=self.is_pocketed
        )
    
    @property
    def speed(self) -> float:
        return np.linalg.norm(self.velocity)
    
    @property
    def kinetic_energy(self) -> float:
        return 0.5 * self.mass * self.speed ** 2
    
    @property
    def momentum(self) -> np.ndarray:
        return self.mass * self.velocity


@dataclass
class Pocket:
    """A pocket on the table."""
    id: int
    position: np.ndarray  # [x, y] center
    radius: float


@dataclass
class TableState:
    """Complete state of the table at a given time."""
    time: float
    balls: List[Ball]
    tilt_roll: float  # θ_x rotation around x-axis (tilts in y direction)
    tilt_pitch: float  # θ_y rotation around y-axis (tilts in x direction)
    balls_pocketed: int = 0
    
    def copy(self) -> 'TableState':
        return TableState(
            time=self.time,
            balls=[b.copy() for b in self.balls],
            tilt_roll=self.tilt_roll,
            tilt_pitch=self.tilt_pitch,
            balls_pocketed=self.balls_pocketed
        )
    
    @property
    def n_balls_remaining(self) -> int:
        return sum(1 for b in self.balls if not b.is_pocketed)
    
    @property
    def active_balls(self) -> List[Ball]:
        return [b for b in self.balls if not b.is_pocketed]
    
    def get_positions(self) -> np.ndarray:
        """Get positions of active balls as (N, 2) array."""
        return np.array([b.position for b in self.active_balls])
    
    def get_velocities(self) -> np.ndarray:
        """Get velocities of active balls as (N, 2) array."""
        return np.array([b.velocity for b in self.active_balls])


# =============================================================================
# PHYSICS ENGINE
# =============================================================================

class CollisionType(Enum):
    BALL_BALL = "ball_ball"
    BALL_WALL = "ball_wall"
    BALL_POCKET = "ball_pocket"


@dataclass
class CollisionEvent:
    """Record of a collision."""
    time: float
    collision_type: CollisionType
    ball_id: int
    other_id: Optional[int] = None  # Other ball ID for ball-ball
    position: Optional[np.ndarray] = None
    impulse_magnitude: float = 0.0


class TiltingTablePhysics:
    """
    Physics simulation for the tilting billiards table.
    
    Handles:
    - Gravity-induced acceleration from tilt
    - Rolling friction
    - Ball-ball elastic collisions
    - Ball-wall reflections
    - Pocket detection
    """
    
    def __init__(self, config: TableConfig):
        self.config = config
        self.pockets = self._create_pockets()
        self.collision_log: List[CollisionEvent] = []
    
    def _create_pockets(self) -> List[Pocket]:
        """Create 6 pockets: 4 corners + 2 side pockets."""
        cfg = self.config
        pockets = []
        
        # Corner pockets
        corners = [
            (-cfg.half_length, -cfg.half_width),
            (-cfg.half_length, cfg.half_width),
            (cfg.half_length, -cfg.half_width),
            (cfg.half_length, cfg.half_width),
        ]
        for i, (x, y) in enumerate(corners):
            pockets.append(Pocket(
                id=i,
                position=np.array([x, y]),
                radius=cfg.pocket_radius
            ))
        
        # Side pockets (middle of long sides)
        sides = [
            (0, -cfg.half_width),
            (0, cfg.half_width),
        ]
        for i, (x, y) in enumerate(sides):
            pockets.append(Pocket(
                id=4 + i,
                position=np.array([x, y]),
                radius=cfg.pocket_radius * 0.9  # Side pockets slightly smaller
            ))
        
        return pockets
    
    def compute_gravity_acceleration(
        self,
        tilt_roll: float,
        tilt_pitch: float
    ) -> np.ndarray:
        """
        Compute acceleration due to gravity given table tilt.
        
        Roll (θ_x): Rotation around x-axis → gravity component in y
        Pitch (θ_y): Rotation around y-axis → gravity component in x
        """
        g = self.config.gravity
        
        # Small angle approximation: sin(θ) ≈ θ for θ < 10°
        ax = g * np.sin(tilt_pitch)  # Pitch tilts table in x direction
        ay = g * np.sin(tilt_roll)   # Roll tilts table in y direction
        
        return np.array([ax, ay])
    
    def compute_friction(self, ball: Ball) -> np.ndarray:
        """Compute rolling friction deceleration."""
        speed = ball.speed
        if speed < 1e-6:
            return np.zeros(2)
        
        # Rolling friction opposes motion
        friction_mag = self.config.rolling_friction * self.config.gravity
        friction_dir = -ball.velocity / speed
        
        return friction_mag * friction_dir
    
    def check_ball_ball_collision(self, b1: Ball, b2: Ball) -> bool:
        """Check if two balls are colliding."""
        dist = np.linalg.norm(b1.position - b2.position)
        min_dist = b1.radius + b2.radius
        return dist <= min_dist
    
    def resolve_ball_ball_collision(
        self,
        b1: Ball,
        b2: Ball,
        restitution: float
    ) -> float:
        """
        Resolve elastic collision between two balls.
        Returns impulse magnitude.
        """
        # Vector from b1 to b2
        delta = b2.position - b1.position
        dist = np.linalg.norm(delta)
        
        if dist < 1e-6:
            return 0.0
        
        # Normal vector
        n = delta / dist
        
        # Relative velocity along collision normal
        v_rel = b1.velocity - b2.velocity
        v_rel_n = np.dot(v_rel, n)
        
        # Only resolve if approaching
        if v_rel_n <= 0:
            return 0.0
        
        # Impulse magnitude (1D collision formula)
        m1, m2 = b1.mass, b2.mass
        j = (1 + restitution) * v_rel_n / (1/m1 + 1/m2)
        
        # Apply impulse
        b1.velocity -= (j / m1) * n
        b2.velocity += (j / m2) * n
        
        # Separate balls to prevent overlap
        overlap = (b1.radius + b2.radius) - dist
        if overlap > 0:
            separation = (overlap / 2 + 0.001) * n
            b1.position -= separation
            b2.position += separation
        
        return j
    
    def check_wall_collision(self, ball: Ball) -> Tuple[bool, str]:
        """Check if ball is colliding with a wall."""
        cfg = self.config
        x, y = ball.position
        r = ball.radius
        
        if x - r <= -cfg.half_length:
            return True, 'left'
        if x + r >= cfg.half_length:
            return True, 'right'
        if y - r <= -cfg.half_width:
            return True, 'bottom'
        if y + r >= cfg.half_width:
            return True, 'top'
        
        return False, ''
    
    def resolve_wall_collision(self, ball: Ball, wall: str):
        """Resolve ball-wall collision."""
        cfg = self.config
        r = ball.radius
        restitution = cfg.wall_restitution
        
        if wall == 'left':
            ball.position[0] = -cfg.half_length + r
            ball.velocity[0] = -ball.velocity[0] * restitution
        elif wall == 'right':
            ball.position[0] = cfg.half_length - r
            ball.velocity[0] = -ball.velocity[0] * restitution
        elif wall == 'bottom':
            ball.position[1] = -cfg.half_width + r
            ball.velocity[1] = -ball.velocity[1] * restitution
        elif wall == 'top':
            ball.position[1] = cfg.half_width - r
            ball.velocity[1] = -ball.velocity[1] * restitution
    
    def check_pocket(self, ball: Ball) -> Optional[int]:
        """Check if ball is in a pocket. Returns pocket ID or None."""
        for pocket in self.pockets:
            dist = np.linalg.norm(ball.position - pocket.position)
            if dist < pocket.radius:
                return pocket.id
        return None
    
    def step(
        self,
        state: TableState,
        action: np.ndarray,
        dt: Optional[float] = None
    ) -> Tuple[TableState, List[CollisionEvent]]:
        """
        Advance simulation by one timestep.
        
        Args:
            state: Current table state
            action: [roll_rate, pitch_rate] in rad/s
            dt: Timestep (uses config default if None)
            
        Returns:
            New state and list of collision events
        """
        if dt is None:
            dt = self.config.dt
        
        cfg = self.config
        new_state = state.copy()
        events = []
        
        # Update tilt angles (with rate and angle limits)
        roll_rate, pitch_rate = action
        roll_rate = np.clip(roll_rate, -cfg.max_tilt_rate, cfg.max_tilt_rate)
        pitch_rate = np.clip(pitch_rate, -cfg.max_tilt_rate, cfg.max_tilt_rate)
        
        new_state.tilt_roll = np.clip(
            state.tilt_roll + roll_rate * dt,
            -cfg.max_tilt_angle,
            cfg.max_tilt_angle
        )
        new_state.tilt_pitch = np.clip(
            state.tilt_pitch + pitch_rate * dt,
            -cfg.max_tilt_angle,
            cfg.max_tilt_angle
        )
        
        # Compute accelerations
        gravity_acc = self.compute_gravity_acceleration(
            new_state.tilt_roll,
            new_state.tilt_pitch
        )
        
        # Update each ball
        for ball in new_state.balls:
            if ball.is_pocketed:
                continue
            
            # Friction
            friction_acc = self.compute_friction(ball)
            
            # Total acceleration
            total_acc = gravity_acc + friction_acc
            
            # Semi-implicit Euler integration
            ball.velocity = ball.velocity + total_acc * dt
            ball.position = ball.position + ball.velocity * dt
            
            # Apply minimum speed threshold
            if ball.speed < 0.001:
                ball.velocity = np.zeros(2)
        
        # Check ball-ball collisions
        active = new_state.active_balls
        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                if self.check_ball_ball_collision(active[i], active[j]):
                    impulse = self.resolve_ball_ball_collision(
                        active[i],
                        active[j],
                        cfg.collision_restitution
                    )
                    if impulse > 0:
                        events.append(CollisionEvent(
                            time=new_state.time + dt,
                            collision_type=CollisionType.BALL_BALL,
                            ball_id=active[i].id,
                            other_id=active[j].id,
                            position=((active[i].position + active[j].position) / 2).copy(),
                            impulse_magnitude=impulse
                        ))
        
        # Check wall collisions
        for ball in new_state.active_balls:
            is_wall, wall = self.check_wall_collision(ball)
            if is_wall:
                self.resolve_wall_collision(ball, wall)
                events.append(CollisionEvent(
                    time=new_state.time + dt,
                    collision_type=CollisionType.BALL_WALL,
                    ball_id=ball.id,
                    position=ball.position.copy()
                ))
        
        # Check pockets
        for ball in new_state.balls:
            if ball.is_pocketed:
                continue
            pocket_id = self.check_pocket(ball)
            if pocket_id is not None:
                ball.is_pocketed = True
                new_state.balls_pocketed += 1
                events.append(CollisionEvent(
                    time=new_state.time + dt,
                    collision_type=CollisionType.BALL_POCKET,
                    ball_id=ball.id,
                    other_id=pocket_id,
                    position=ball.position.copy()
                ))
        
        new_state.time += dt
        self.collision_log.extend(events)
        
        return new_state, events


# =============================================================================
# ENVIRONMENT WRAPPER
# =============================================================================

class TiltingBilliardsEnv:
    """
    Gym-like environment for the tilting billiards table.
    
    Observation: Ball positions and velocities + tilt angles
    Action: Roll and pitch rates
    Reward: +1 for each ball pocketed
    """
    
    def __init__(
        self,
        config: Optional[TableConfig] = None,
        n_balls: int = 7,
        random_seed: Optional[int] = None
    ):
        self.config = config or TableConfig()
        self.n_balls = n_balls
        self.rng = np.random.default_rng(random_seed)
        
        self.physics = TiltingTablePhysics(self.config)
        self.state: Optional[TableState] = None
        
        # For rendering
        self.trajectory_history: List[np.ndarray] = []
    
    def reset(
        self,
        initial_positions: Optional[np.ndarray] = None,
        initial_velocities: Optional[np.ndarray] = None
    ) -> TableState:
        """
        Reset environment with random or specified initial conditions.
        
        Args:
            initial_positions: (n_balls, 2) array of positions
            initial_velocities: (n_balls, 2) array of velocities
            
        Returns:
            Initial state
        """
        cfg = self.config
        
        # Generate balls
        balls = []
        
        if initial_positions is None:
            # Random positions (avoid pockets and overlaps)
            positions = []
            for i in range(self.n_balls):
                for attempt in range(100):
                    x = self.rng.uniform(-cfg.half_length * 0.8, cfg.half_length * 0.8)
                    y = self.rng.uniform(-cfg.half_width * 0.8, cfg.half_width * 0.8)
                    pos = np.array([x, y])
                    
                    # Check distance from pockets
                    too_close_to_pocket = any(
                        np.linalg.norm(pos - p.position) < p.radius + cfg.ball_radius * 2
                        for p in self.physics.pockets
                    )
                    
                    # Check distance from other balls
                    too_close_to_ball = any(
                        np.linalg.norm(pos - other) < cfg.ball_radius * 3
                        for other in positions
                    )
                    
                    if not too_close_to_pocket and not too_close_to_ball:
                        positions.append(pos)
                        break
                else:
                    # Fallback: just place it
                    positions.append(np.array([x, y]))
            
            initial_positions = np.array(positions)
        
        if initial_velocities is None:
            # Random velocities (moderate speed)
            speeds = self.rng.uniform(0.1, 0.5, self.n_balls)
            angles = self.rng.uniform(0, 2 * np.pi, self.n_balls)
            initial_velocities = np.stack([
                speeds * np.cos(angles),
                speeds * np.sin(angles)
            ], axis=1)
        
        for i in range(self.n_balls):
            mass = cfg.ball_mass_mean + self.rng.normal(0, cfg.ball_mass_std)
            mass = max(0.1, mass)  # Ensure positive
            
            balls.append(Ball(
                id=i,
                position=initial_positions[i].copy(),
                velocity=initial_velocities[i].copy(),
                mass=mass,
                radius=cfg.ball_radius
            ))
        
        self.state = TableState(
            time=0.0,
            balls=balls,
            tilt_roll=0.0,
            tilt_pitch=0.0,
            balls_pocketed=0
        )
        
        self.physics.collision_log = []
        self.trajectory_history = [self.state.get_positions().copy()]
        
        return self.state.copy()
    
    def step(
        self,
        action: np.ndarray,
        n_substeps: int = 10
    ) -> Tuple[TableState, float, bool, Dict]:
        """
        Take an action and advance the environment.
        
        Args:
            action: [roll_rate, pitch_rate] in rad/s
            n_substeps: Number of physics substeps per env step
            
        Returns:
            state, reward, done, info
        """
        if self.state is None:
            raise RuntimeError("Must call reset() before step()")
        
        action = np.array(action)
        prev_pocketed = self.state.balls_pocketed
        all_events = []
        
        # Run physics substeps
        for _ in range(n_substeps):
            self.state, events = self.physics.step(self.state, action)
            all_events.extend(events)
        
        # Record trajectory
        if self.state.n_balls_remaining > 0:
            self.trajectory_history.append(self.state.get_positions().copy())
        
        # Compute reward
        new_pocketed = self.state.balls_pocketed - prev_pocketed
        reward = float(new_pocketed)
        
        # Check done
        done = self.state.n_balls_remaining == 0
        
        info = {
            'balls_pocketed': self.state.balls_pocketed,
            'balls_remaining': self.state.n_balls_remaining,
            'collision_events': all_events,
            'n_ball_ball_collisions': sum(
                1 for e in all_events if e.collision_type == CollisionType.BALL_BALL
            )
        }
        
        return self.state.copy(), reward, done, info
    
    def compute_density_field(
        self,
        state: Optional[TableState] = None,
        resolution: int = 20
    ) -> np.ndarray:
        """
        Compute 2D density field for active balls.
        
        This is what the FNO will learn to predict!
        
        Args:
            state: State to compute density for (uses current if None)
            resolution: Grid resolution per dimension
            
        Returns:
            (resolution, resolution) density array
        """
        if state is None:
            state = self.state
        
        cfg = self.config
        
        # Create grid
        x_edges = np.linspace(-cfg.half_length, cfg.half_length, resolution + 1)
        y_edges = np.linspace(-cfg.half_width, cfg.half_width, resolution + 1)
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2

        # Vectorized meshgrid: shape (resolution, resolution)
        XX, YY = np.meshgrid(x_centers, y_centers)
        density = np.zeros((resolution, resolution))

        sigma = cfg.ball_radius * 2  # Smoothing width
        inv_two_sigma_sq = 1.0 / (2.0 * sigma ** 2)

        for ball in state.active_balls:
            dist_sq = (XX - ball.position[0]) ** 2 + (YY - ball.position[1]) ** 2
            density += np.exp(-dist_sq * inv_two_sigma_sq)

        return density
    
    def get_observation(self) -> Dict[str, np.ndarray]:
        """Get observation dict for RL/MPC."""
        state = self.state
        
        # Pad to fixed size for neural network input
        max_balls = self.n_balls
        positions = np.zeros((max_balls, 2))
        velocities = np.zeros((max_balls, 2))
        active_mask = np.zeros(max_balls)
        
        for i, ball in enumerate(state.balls):
            if not ball.is_pocketed:
                positions[i] = ball.position
                velocities[i] = ball.velocity
                active_mask[i] = 1.0
        
        return {
            'positions': positions,
            'velocities': velocities,
            'active_mask': active_mask,
            'tilt': np.array([state.tilt_roll, state.tilt_pitch]),
            'density': self.compute_density_field(state)
        }
    
    def estimate_lyapunov_exponent(
        self,
        perturbation: float = 1e-6,
        duration: float = 2.0,
        n_trials: int = 5
    ) -> float:
        """
        Estimate Lyapunov exponent for current configuration.
        
        This shows the system is chaotic!
        """
        cfg = self.config
        lyapunovs = []
        
        for trial in range(n_trials):
            # Reset to same initial state
            state_original = self.state.copy()
            
            # Create perturbed state
            state_perturbed = self.state.copy()
            if state_perturbed.active_balls:
                state_perturbed.active_balls[0].position[0] += perturbation
            
            # Evolve both
            physics1 = TiltingTablePhysics(cfg)
            physics2 = TiltingTablePhysics(cfg)
            
            n_steps = int(duration / cfg.dt)
            action = np.zeros(2)  # No tilt
            
            for _ in range(n_steps):
                state_original, _ = physics1.step(state_original, action)
                state_perturbed, _ = physics2.step(state_perturbed, action)
            
            # Compute divergence
            pos1 = state_original.get_positions()
            pos2 = state_perturbed.get_positions()
            
            if len(pos1) > 0 and len(pos2) > 0 and len(pos1) == len(pos2):
                divergence = np.sqrt(np.sum((pos1 - pos2)**2))
                if divergence > perturbation:
                    lyap = np.log(divergence / perturbation) / duration
                    lyapunovs.append(lyap)
        
        return np.mean(lyapunovs) if lyapunovs else 0.0


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_random_scenario(
    n_balls: int = 7,
    seed: Optional[int] = None,
    difficulty: str = 'medium'
) -> TiltingBilliardsEnv:
    """
    Create environment with random initial conditions.
    
    Args:
        n_balls: Number of balls
        seed: Random seed
        difficulty: 'easy', 'medium', or 'hard'
            - easy: Balls start slow, clustered
            - medium: Moderate speeds, spread out
            - hard: Fast balls, chaotic initial conditions
    """
    config = TableConfig()
    env = TiltingBilliardsEnv(config, n_balls=n_balls, random_seed=seed)
    
    rng = np.random.default_rng(seed)
    
    if difficulty == 'easy':
        # Clustered near center, slow
        positions = rng.normal(0, 0.1, (n_balls, 2))
        positions = np.clip(positions, -0.3, 0.3)
        velocities = rng.uniform(-0.1, 0.1, (n_balls, 2))
    
    elif difficulty == 'medium':
        positions = None  # Use default random
        velocities = None
    
    elif difficulty == 'hard':
        # Spread out, fast, more chaotic
        positions = rng.uniform(-0.4, 0.4, (n_balls, 2))
        speeds = rng.uniform(0.3, 0.8, n_balls)
        angles = rng.uniform(0, 2 * np.pi, n_balls)
        velocities = np.stack([
            speeds * np.cos(angles),
            speeds * np.sin(angles)
        ], axis=1)
    
    else:
        raise ValueError(f"Unknown difficulty: {difficulty}")
    
    env.reset(initial_positions=positions, initial_velocities=velocities)
    return env


def run_episode(
    env: TiltingBilliardsEnv,
    controller,  # Callable: observation -> action
    max_steps: int = 1000,
    render_interval: int = 10
) -> Dict:
    """
    Run a full episode with a controller.
    
    Returns episode statistics.
    """
    state = env.reset() if env.state is None else env.state
    
    total_reward = 0
    trajectory = [state.copy()]
    actions = []
    
    for step in range(max_steps):
        obs = env.get_observation()
        action = controller(obs)
        
        state, reward, done, info = env.step(action)
        
        total_reward += reward
        trajectory.append(state.copy())
        actions.append(action)
        
        if done:
            break
    
    return {
        'total_reward': total_reward,
        'balls_pocketed': state.balls_pocketed,
        'steps': len(trajectory) - 1,
        'final_state': state,
        'trajectory': trajectory,
        'actions': actions
    }
