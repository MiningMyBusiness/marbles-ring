"""
Core Particle System for 1D Hard-Sphere Dynamics

This module implements the fundamental physics of a 1D hard-sphere gas
on a circular track. It serves as the foundation for both:
- Paper 1: Neural operator methodology (abstract hard spheres)
- Paper 2: Polyribosome biological application

Physics:
- Perfectly elastic collisions (energy conserving)
- Deterministic dynamics (no stochastic terms at this level)
- Single-file diffusion (particles cannot pass)
- Circular (periodic) boundary conditions

Key insight: With unequal masses (N >= 3), this system is chaotic with
positive Lyapunov exponents, making long-term trajectory prediction
impossible but density evolution learnable.

Author: Kiran Bhattacharyya
License: MIT
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Callable
from enum import Enum
import heapq


class CollisionType(Enum):
    """Types of collision events in the system."""
    PARTICLE_PARTICLE = "particle_particle"
    # Future: could add wall collisions for non-periodic systems


@dataclass
class Particle:
    """
    A single particle (hard sphere) in the 1D system.
    
    In the biological interpretation:
    - position: location along mRNA (in nm or codons)
    - velocity: translation speed (codons/second or nm/s)
    - diameter: ribosome footprint + nascent chain exclusion volume
    - mass: effective hydrodynamic mass (ribosome + nascent chain)
    
    Attributes:
        id: Unique identifier
        position: Position along track [0, L)
        velocity: Signed velocity (positive = increasing position)
        diameter: Particle diameter (as fraction of track length or absolute)
        mass: Particle mass (determines collision dynamics)
        metadata: Optional dict for biological annotations
    """
    id: int
    position: float
    velocity: float
    diameter: float
    mass: float
    metadata: dict = field(default_factory=dict)
    
    @property
    def momentum(self) -> float:
        """Linear momentum p = mv."""
        return self.mass * self.velocity
    
    @property
    def kinetic_energy(self) -> float:
        """Kinetic energy E = 0.5 * m * v^2."""
        return 0.5 * self.mass * self.velocity ** 2
    
    @property
    def radius(self) -> float:
        """Half-diameter for collision calculations."""
        return self.diameter / 2.0
    
    def copy(self) -> 'Particle':
        """Create a deep copy of this particle."""
        return Particle(
            id=self.id,
            position=self.position,
            velocity=self.velocity,
            diameter=self.diameter,
            mass=self.mass,
            metadata=self.metadata.copy()
        )


@dataclass
class CollisionEvent:
    """
    Record of a collision event.
    
    These events form the "collision graph" that encodes system dynamics.
    For the inverse problem (Paper 1), we attempt to infer particle
    properties from sequences of these events.
    """
    time: float
    particle_i: int
    particle_j: int
    position: float  # Where on the track the collision occurred
    
    # Pre-collision velocities
    v_i_before: float
    v_j_before: float
    
    # Post-collision velocities
    v_i_after: float
    v_j_after: float
    
    # Momentum transfer
    delta_p: float = field(init=False)
    
    def __post_init__(self):
        # This is what the "blind agent" would feel
        self.delta_p = abs(self.v_i_after - self.v_i_before)


@dataclass 
class SystemState:
    """
    Complete microstate of the system at a given time.
    
    This is what we're trying to predict (Paper 1) or use to
    understand biological function (Paper 2).
    """
    time: float
    particles: List[Particle]
    
    @property
    def positions(self) -> np.ndarray:
        return np.array([p.position for p in self.particles])
    
    @property
    def velocities(self) -> np.ndarray:
        return np.array([p.velocity for p in self.particles])
    
    @property
    def masses(self) -> np.ndarray:
        return np.array([p.mass for p in self.particles])
    
    @property
    def diameters(self) -> np.ndarray:
        return np.array([p.diameter for p in self.particles])
    
    @property
    def total_momentum(self) -> float:
        return sum(p.momentum for p in self.particles)
    
    @property
    def total_energy(self) -> float:
        return sum(p.kinetic_energy for p in self.particles)
    
    def copy(self) -> 'SystemState':
        return SystemState(
            time=self.time,
            particles=[p.copy() for p in self.particles]
        )


class CircularTrack:
    """
    A circular (periodic) 1D track of length L.
    
    Handles:
    - Position wrapping
    - Distance calculations respecting periodicity
    - Collision detection between adjacent particles
    
    In biological terms, this represents a circularized mRNA
    where the 5' and 3' ends are bridged by eIF4E-PABP interactions.
    """
    
    def __init__(self, length: float):
        """
        Initialize track.
        
        Args:
            length: Total track length. For mRNA, this might be
                   in nanometers or codon units.
        """
        self.length = length
    
    def wrap_position(self, position: float) -> float:
        """Wrap position to [0, L)."""
        return position % self.length
    
    def signed_distance(self, pos1: float, pos2: float) -> float:
        """
        Compute signed distance from pos1 to pos2.
        
        Returns the shortest path distance, positive if pos2 is
        "ahead" of pos1 in the positive direction.
        """
        diff = pos2 - pos1
        if diff > self.length / 2:
            diff -= self.length
        elif diff < -self.length / 2:
            diff += self.length
        return diff
    
    def distance(self, pos1: float, pos2: float) -> float:
        """Unsigned distance between two positions."""
        return abs(self.signed_distance(pos1, pos2))
    
    def are_adjacent(self, particles: List[Particle], i: int, j: int) -> bool:
        """
        Check if particles i and j are adjacent in the ordering.
        
        In single-file diffusion, only adjacent particles can collide.
        """
        # Sort particles by position to determine adjacency
        sorted_indices = np.argsort([p.position for p in particles])
        pos_i = np.where(sorted_indices == i)[0][0]
        pos_j = np.where(sorted_indices == j)[0][0]
        
        n = len(particles)
        return (pos_j - pos_i) % n == 1 or (pos_i - pos_j) % n == 1


class CollisionEngine:
    """
    Handles collision detection and resolution.
    
    Uses event-driven simulation: instead of small timesteps,
    we compute exact collision times and jump between events.
    This is crucial for:
    1. Numerical accuracy (no discretization error)
    2. Generating clean collision timestamp data for ML
    """
    
    def __init__(self, track: CircularTrack, tolerance: float = 1e-10):
        self.track = track
        self.tolerance = tolerance
    
    def time_to_collision(self, p1: Particle, p2: Particle) -> Optional[float]:
        """
        Compute time until particles p1 and p2 collide.
        
        Returns None if they won't collide (moving apart or parallel).
        
        Math: On a periodic track, we need the particles to be at
        distance (r1 + r2) with approaching velocities.
        """
        # Signed distance from p1 to p2.
        # np.real() safely strips any imaginary component from numpy complex
        # subtypes (np.complex128 etc.) that isinstance(..., complex) misses.
        _dx = self.track.signed_distance(p1.position, p2.position)
        dx  = float(np.real(_dx))

        # Relative velocity (positive if p1 approaching p2)
        dv = float(np.real(p1.velocity)) - float(np.real(p2.velocity))

        # Contact distance
        contact_dist = p1.radius + p2.radius

        # Current separation (surface to surface)
        separation = abs(dx) - contact_dist

        if separation < -self.tolerance:
            # Already overlapping - this shouldn't happen in well-behaved simulation
            return 0.0
        
        # They collide when |dx - dv*t| = contact_dist
        # We want the first positive t where they meet
        
        if dx > 0:
            # p2 is ahead of p1 in positive direction
            # Collision if p1 catches up (dv > 0)
            if dv > self.tolerance:
                t = (dx - contact_dist) / dv
                return t if t > self.tolerance else None
        else:
            # p2 is behind p1 (or equivalently, ahead in negative direction)
            # Collision if p2 catches up (dv < 0)
            if dv < -self.tolerance:
                t = (dx + contact_dist) / dv
                return t if t > self.tolerance else None
        
        # Also check wrap-around collision
        if dx > 0:
            # Check if p2 wraps around and hits p1 from behind
            dx_wrap = dx - self.track.length
            if dv < -self.tolerance:
                t = (dx_wrap + contact_dist) / dv
                return t if t > self.tolerance else None
        else:
            dx_wrap = dx + self.track.length
            if dv > self.tolerance:
                t = (dx_wrap - contact_dist) / dv
                return t if t > self.tolerance else None
        
        return None
    
    def resolve_collision(self, p1: Particle, p2: Particle) -> Tuple[float, float]:
        """
        Resolve elastic collision between two particles.
        
        Uses 1D elastic collision formulas:
        v1' = ((m1-m2)*v1 + 2*m2*v2) / (m1+m2)
        v2' = ((m2-m1)*v2 + 2*m1*v1) / (m1+m2)
        
        Returns:
            Tuple of new velocities (v1_new, v2_new)
        """
        m1, m2 = p1.mass, p2.mass
        v1, v2 = p1.velocity, p2.velocity
        
        total_mass = m1 + m2
        
        v1_new = ((m1 - m2) * v1 + 2 * m2 * v2) / total_mass
        v2_new = ((m2 - m1) * v2 + 2 * m1 * v1) / total_mass
        
        return v1_new, v2_new


class ParticleSystem:
    """
    Main simulation class for the 1D hard-sphere gas.
    
    Implements event-driven molecular dynamics on a circular track.
    
    Key outputs:
    - Time series of system states (for density evolution)
    - Collision event log (for inverse problem / ML training)
    - Conserved quantities (for validation)
    
    Usage:
        system = ParticleSystem(track_length=1000.0)
        system.add_particle(position=0, velocity=10, diameter=5, mass=1)
        system.add_particle(position=100, velocity=-5, diameter=8, mass=2)
        
        # Evolve for 100 time units
        states, collisions = system.evolve(duration=100.0, save_interval=1.0)
    """
    
    def __init__(self, track_length: float):
        self.track = CircularTrack(track_length)
        self.collision_engine = CollisionEngine(self.track)
        self.particles: List[Particle] = []
        self.time = 0.0
        self.collision_log: List[CollisionEvent] = []
        self._next_id = 0
    
    def add_particle(
        self,
        position: float,
        velocity: float,
        diameter: float,
        mass: float,
        metadata: Optional[dict] = None
    ) -> Particle:
        """Add a particle to the system."""
        particle = Particle(
            id=self._next_id,
            position=self.track.wrap_position(position),
            velocity=velocity,
            diameter=diameter,
            mass=mass,
            metadata=metadata or {}
        )
        self.particles.append(particle)
        self._next_id += 1
        return particle
    
    def get_state(self) -> SystemState:
        """Get current system state."""
        return SystemState(
            time=self.time,
            particles=[p.copy() for p in self.particles]
        )
    
    def _get_ordered_particles(self) -> List[int]:
        """Get particle indices in position order."""
        return list(np.argsort([p.position for p in self.particles]))
    
    def _find_next_collision(self) -> Optional[Tuple[float, int, int]]:
        """
        Find the next collision event.
        
        Only checks adjacent pairs (single-file constraint).
        
        Returns:
            Tuple of (time, particle_i, particle_j) or None
        """
        ordered = self._get_ordered_particles()
        n = len(ordered)
        
        if n < 2:
            return None
        
        best_time = float('inf')
        best_pair = None
        
        # Check all adjacent pairs (including wrap-around)
        for k in range(n):
            i = ordered[k]
            j = ordered[(k + 1) % n]
            
            t = self.collision_engine.time_to_collision(
                self.particles[i], 
                self.particles[j]
            )
            
            if t is not None and t < best_time:
                best_time = t
                best_pair = (i, j)
        
        if best_pair is None:
            return None
        
        return (best_time, best_pair[0], best_pair[1])
    
    def _advance_particles(self, dt: float):
        """Move all particles forward by dt."""
        for p in self.particles:
            # Force position to a real float — Lyapunov perturbation arithmetic
            # can occasionally introduce a tiny imaginary component via numpy,
            # which would propagate and crash comparison operators downstream.
            raw = p.position + float(np.real(p.velocity)) * dt
            p.position = self.track.wrap_position(float(np.real(raw)))
        self.time += dt
    
    def _process_collision(self, i: int, j: int) -> CollisionEvent:
        """Process collision between particles i and j."""
        p_i = self.particles[i]
        p_j = self.particles[j]
        
        v_i_before = p_i.velocity
        v_j_before = p_j.velocity
        
        v_i_after, v_j_after = self.collision_engine.resolve_collision(p_i, p_j)
        
        p_i.velocity = v_i_after
        p_j.velocity = v_j_after
        
        # Record collision
        event = CollisionEvent(
            time=self.time,
            particle_i=i,
            particle_j=j,
            position=p_i.position,
            v_i_before=v_i_before,
            v_j_before=v_j_before,
            v_i_after=v_i_after,
            v_j_after=v_j_after
        )
        self.collision_log.append(event)
        
        return event
    
    def evolve(
        self,
        duration: float,
        save_interval: float = 0.1,
        max_collisions: int = 1000000,
        callback: Optional[Callable[[SystemState, Optional[CollisionEvent]], None]] = None
    ) -> Tuple[List[SystemState], List[CollisionEvent]]:
        """
        Evolve the system for a given duration.
        
        Uses event-driven simulation:
        1. Find next collision time
        2. Advance all particles to that time
        3. Resolve collision
        4. Repeat
        
        Args:
            duration: Total time to simulate
            save_interval: How often to save state snapshots
            max_collisions: Safety limit on number of collisions
            callback: Optional function called after each event
            
        Returns:
            Tuple of (list of states, list of collision events)
        """
        states = [self.get_state()]
        collisions_in_window = []
        
        target_time = self.time + duration
        next_save = self.time + save_interval
        collision_count = 0
        
        while self.time < target_time and collision_count < max_collisions:
            # Find next collision
            next_collision = self._find_next_collision()
            
            if next_collision is None:
                # No more collisions possible - just advance to end
                self._advance_particles(target_time - self.time)
                break
            
            dt, i, j = next_collision
            
            # Check if collision happens before target time
            if self.time + dt > target_time:
                self._advance_particles(target_time - self.time)
                break
            
            # Save states at regular intervals while advancing
            while next_save < self.time + dt and next_save < target_time:
                advance_to_save = next_save - self.time
                self._advance_particles(advance_to_save)
                states.append(self.get_state())
                dt -= advance_to_save
                next_save += save_interval
            
            # Advance to collision
            self._advance_particles(dt)
            
            # Process collision
            event = self._process_collision(i, j)
            collisions_in_window.append(event)
            collision_count += 1
            
            if callback:
                callback(self.get_state(), event)
        
        # Final state
        if states[-1].time < self.time:
            states.append(self.get_state())
        
        return states, collisions_in_window
    
    def compute_density_field(
        self,
        n_bins: int = 100,
        kernel_width: Optional[float] = None
    ) -> np.ndarray:
        """
        Compute coarse-grained density field.
        
        This is the key observable for:
        - Paper 1: What the neural operator predicts
        - Paper 2: What we compare to ribosome profiling data
        
        Args:
            n_bins: Number of spatial bins
            kernel_width: Gaussian smoothing width (default: 2*max_diameter)
            
        Returns:
            1D array of density values
        """
        if kernel_width is None:
            kernel_width = 2 * max(p.diameter for p in self.particles)
        
        bin_edges = np.linspace(0, self.track.length, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        density = np.zeros(n_bins)
        
        for p in self.particles:
            # Gaussian kernel centered at particle position
            for k, center in enumerate(bin_centers):
                dist = self.track.distance(p.position, center)
                density[k] += np.exp(-0.5 * (dist / kernel_width) ** 2)
        
        # Normalize
        density /= (kernel_width * np.sqrt(2 * np.pi))
        
        return density
    
    def compute_lyapunov_exponent(
        self,
        perturbation: float = 1e-8,
        duration: float = 100.0
    ) -> float:
        """
        Estimate maximum Lyapunov exponent by trajectory divergence.
        
        This quantifies the chaos: larger λ means faster divergence
        and shorter Lyapunov time T_lyap ≈ 1/λ.
        
        Args:
            perturbation: Initial perturbation magnitude
            duration: Time over which to measure divergence
            
        Returns:
            Estimated Lyapunov exponent
        """
        # Save current state
        original_state = self.get_state()
        original_time = self.time
        original_log = self.collision_log.copy()
        
        # Evolve original trajectory
        self.evolve(duration)
        final_positions = np.array([p.position for p in self.particles])
        
        # Reset
        self.time = original_time
        self.collision_log = original_log
        for i, p in enumerate(self.particles):
            orig = original_state.particles[i]
            p.position = orig.position
            p.velocity = orig.velocity
        
        # Perturb one particle's position
        self.particles[0].position += perturbation
        
        # Evolve perturbed trajectory
        self.evolve(duration)
        perturbed_positions = np.array([p.position for p in self.particles])
        
        # Compute divergence
        divergence = np.sqrt(np.sum(
            [self.track.distance(final_positions[i], perturbed_positions[i])**2 
             for i in range(len(self.particles))]
        ))
        
        # Reset to original
        self.time = original_time
        self.collision_log = original_log
        for i, p in enumerate(self.particles):
            orig = original_state.particles[i]
            p.position = orig.position
            p.velocity = orig.velocity
        
        # λ ≈ (1/t) * ln(divergence / perturbation)
        lyapunov = np.log(divergence / perturbation) / duration
        
        return lyapunov


def create_random_system(
    n_particles: int,
    track_length: float = 1000.0,
    diameter_range: Tuple[float, float] = (5.0, 20.0),
    speed_range: Tuple[float, float] = (10.0, 50.0),
    density_dependent_mass: bool = True,
    seed: Optional[int] = None
) -> ParticleSystem:
    """
    Create a system with randomly initialized particles.
    
    Args:
        n_particles: Number of particles
        track_length: Length of circular track
        diameter_range: (min, max) diameter
        speed_range: (min, max) speed magnitude
        density_dependent_mass: If True, mass ∝ diameter³ (equal density)
        seed: Random seed for reproducibility
        
    Returns:
        Initialized ParticleSystem
    """
    if seed is not None:
        np.random.seed(seed)
    
    system = ParticleSystem(track_length)
    
    # Generate diameters
    diameters = np.random.uniform(
        diameter_range[0], 
        diameter_range[1], 
        n_particles
    )
    
    # Ensure particles fit on track with spacing
    total_diameter = np.sum(diameters)
    if total_diameter > 0.5 * track_length:
        scale = 0.4 * track_length / total_diameter
        diameters *= scale
    
    # Generate non-overlapping positions
    spacing = track_length / n_particles
    positions = np.zeros(n_particles)
    
    cumulative = 0
    for i in range(n_particles):
        min_pos = cumulative + diameters[i] / 2
        max_pos = cumulative + spacing - diameters[(i+1) % n_particles] / 2
        positions[i] = np.random.uniform(min_pos, max_pos)
        cumulative += spacing
    
    # Generate velocities
    speeds = np.random.uniform(speed_range[0], speed_range[1], n_particles)
    directions = np.random.choice([-1, 1], n_particles)
    velocities = speeds * directions
    
    # Add particles
    for i in range(n_particles):
        mass = diameters[i] ** 3 if density_dependent_mass else 1.0
        system.add_particle(
            position=positions[i],
            velocity=velocities[i],
            diameter=diameters[i],
            mass=mass
        )
    
    return system


# Validation functions
def validate_conservation(states: List[SystemState], tolerance: float = 1e-6) -> dict:
    """Check that energy and momentum are conserved."""
    energies = [s.total_energy for s in states]
    momenta = [s.total_momentum for s in states]
    
    energy_drift = max(energies) - min(energies)
    momentum_drift = max(momenta) - min(momenta)
    
    return {
        'energy_conserved': energy_drift < tolerance,
        'momentum_conserved': momentum_drift < tolerance,
        'energy_drift': energy_drift,
        'momentum_drift': momentum_drift,
        'initial_energy': energies[0],
        'initial_momentum': momenta[0]
    }
