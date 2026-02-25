"""
Biological Parameterization: Ribosome and mRNA Models

This module maps the abstract hard-sphere system to polyribosome biology.

Key biological facts encoded:
1. Ribosomes are ~25-30 nm in diameter
2. They protect ~28-30 nucleotides of mRNA footprint
3. mRNA codons are ~3 nucleotides, so footprint ≈ 9-10 codons
4. Translation speed: ~5-20 codons/second (varies by codon, tRNA availability)
5. Nascent polypeptide chains grow as translation proceeds
6. Chain length affects hydrodynamic drag (effective mass)
7. Chain folding affects excluded volume (effective diameter)

This is the core innovation for Paper 2: treating ribosomes as particles
whose mass and diameter CHANGE as they translate, creating the 
heterogeneous chaotic system.

References:
- Ingolia et al. (2009) Science - Ribosome profiling
- Wolin & Walter (1988) EMBO J - Ribosome footprinting
- Sharma et al. (2019) Nature - Nascent chain folding dynamics

Author: Kiran Bhattacharyya
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from enum import Enum
import json

from .particle_system import Particle, ParticleSystem, CircularTrack


# =============================================================================
# Physical Constants (SI units, then converted to simulation units)
# =============================================================================

@dataclass(frozen=True)
class BiologicalConstants:
    """
    Literature-derived constants for eukaryotic translation.
    
    Sources are noted for each parameter.
    Simulation uses "codon units" for position and "seconds" for time.
    """
    
    # Ribosome dimensions
    ribosome_diameter_nm: float = 25.0  # ~25-30 nm diameter
    ribosome_footprint_codons: int = 10  # ~28-30 nt ≈ 9-10 codons
    ribosome_mass_MDa: float = 4.2  # Megadaltons (eukaryotic 80S)
    
    # mRNA properties
    codon_spacing_nm: float = 0.9  # ~0.9 nm per codon (assumes extended RNA)
    
    # Translation kinetics
    elongation_rate_codons_per_sec: float = 5.0  # Typical, varies 2-20
    initiation_rate_per_sec: float = 0.1  # Highly variable
    
    # Nascent chain properties
    amino_acid_mass_Da: float = 110.0  # Average amino acid mass
    residue_length_nm: float = 0.38  # Extended polypeptide per residue
    
    # Ribosome exit tunnel
    exit_tunnel_length_residues: int = 40  # ~40 amino acids fit in tunnel
    exit_tunnel_length_nm: float = 10.0  # ~10 nm tunnel
    
    # Hydrodynamic properties
    cytoplasm_viscosity_cP: float = 2.0  # ~2-3 cP, varies
    temperature_K: float = 310.0  # 37°C = 310 K


# Default constants instance
BIO_CONSTANTS = BiologicalConstants()


# =============================================================================
# Nascent Chain Model
# =============================================================================

class FoldingState(Enum):
    """
    Simplified model of nascent chain folding.
    
    Folding affects both hydrodynamic radius (drag/mass) and 
    excluded volume (collision diameter).
    """
    EXTENDED = "extended"      # Random coil, maximum drag
    PARTIALLY_FOLDED = "partial"  # Intermediate
    COMPACT = "compact"         # Globular, minimum drag


@dataclass
class NascentChain:
    """
    Model of the growing polypeptide chain.
    
    As the ribosome translates, the nascent chain:
    1. Grows longer (more amino acids)
    2. May begin to fold (if folding domains are complete)
    3. Increases the effective mass and diameter of the ribosome
    
    This creates the time-dependent heterogeneity that makes
    the polyribosome system chaotic.
    """
    
    # Chain state
    length_residues: int = 0  # Current chain length (amino acids)
    folding_state: FoldingState = FoldingState.EXTENDED
    
    # Properties of the protein being synthesized
    total_length: int = 300  # Full protein length
    folding_domains: List[Tuple[int, int]] = field(default_factory=list)
    # List of (start_residue, end_residue) for folding domains
    
    def grow(self, n_residues: int = 1):
        """Add residues to the chain."""
        self.length_residues = min(
            self.length_residues + n_residues,
            self.total_length
        )
        self._update_folding()
    
    def _update_folding(self):
        """Update folding state based on chain length."""
        # Simple model: fold when complete domains emerge from tunnel
        emerged_length = max(0, self.length_residues - BIO_CONSTANTS.exit_tunnel_length_residues)
        
        if emerged_length == 0:
            self.folding_state = FoldingState.EXTENDED
        elif emerged_length < 50:
            self.folding_state = FoldingState.EXTENDED
        elif emerged_length < 150:
            self.folding_state = FoldingState.PARTIALLY_FOLDED
        else:
            self.folding_state = FoldingState.COMPACT
    
    @property
    def emerged_length(self) -> int:
        """Residues emerged from the ribosome exit tunnel."""
        return max(0, self.length_residues - BIO_CONSTANTS.exit_tunnel_length_residues)
    
    @property
    def mass_contribution_MDa(self) -> float:
        """
        Additional mass from the nascent chain (in MDa).
        
        This adds to the ribosome's base mass.
        """
        mass_Da = self.length_residues * BIO_CONSTANTS.amino_acid_mass_Da
        return mass_Da / 1e6  # Convert to MDa
    
    @property  
    def radius_contribution_nm(self) -> float:
        """
        Additional radius from emerged nascent chain.
        
        Depends on folding state:
        - Extended: R ∝ N^0.6 (self-avoiding walk)
        - Partially folded: R ∝ N^0.5
        - Compact: R ∝ N^0.33 (globular)
        """
        N = self.emerged_length
        if N == 0:
            return 0.0
        
        # Kuhn length for polypeptide ≈ 1 nm
        kuhn_length = 1.0
        
        if self.folding_state == FoldingState.EXTENDED:
            # Self-avoiding walk: R_g ∝ N^0.6
            exponent = 0.6
        elif self.folding_state == FoldingState.PARTIALLY_FOLDED:
            exponent = 0.5
        else:  # COMPACT
            # Globular protein: R ∝ N^0.33
            exponent = 0.33
        
        return kuhn_length * (N ** exponent)


# =============================================================================
# Ribosome Model
# =============================================================================

@dataclass
class Ribosome:
    """
    A ribosome translating an mRNA.
    
    Extends the base Particle with biological properties:
    - Position along mRNA (in codons)
    - Nascent chain (growing polypeptide)
    - Translation state (active, stalled, etc.)
    
    The effective mass and diameter change as translation proceeds,
    creating the heterogeneous system needed for chaos.
    """
    
    # Unique identifier
    id: int
    
    # Position on mRNA (codon units, 0 = start codon)
    position_codons: float
    
    # Translation velocity (codons/second, positive = toward stop codon)
    velocity_codons_per_sec: float
    
    # The growing protein
    nascent_chain: NascentChain = field(default_factory=NascentChain)
    
    # State flags
    is_stalled: bool = False
    stall_reason: Optional[str] = None
    
    # Timing
    time_since_initiation: float = 0.0
    
    @property
    def base_diameter_codons(self) -> float:
        """Ribosome footprint in codon units."""
        return float(BIO_CONSTANTS.ribosome_footprint_codons)
    
    @property
    def effective_diameter_codons(self) -> float:
        """
        Total excluded volume in codon units.
        
        Includes ribosome footprint + emerged nascent chain.
        """
        # Convert nascent chain radius from nm to codons
        chain_radius_nm = self.nascent_chain.radius_contribution_nm
        chain_radius_codons = chain_radius_nm / BIO_CONSTANTS.codon_spacing_nm
        
        # Add chain contribution to both sides
        return self.base_diameter_codons + 2 * chain_radius_codons
    
    @property
    def base_mass(self) -> float:
        """Ribosome mass in MDa."""
        return BIO_CONSTANTS.ribosome_mass_MDa
    
    @property
    def effective_mass(self) -> float:
        """
        Total effective mass including nascent chain.
        
        In MDa (though units cancel in collision dynamics).
        """
        return self.base_mass + self.nascent_chain.mass_contribution_MDa
    
    def to_particle(self) -> Particle:
        """Convert to base Particle class for simulation."""
        return Particle(
            id=self.id,
            position=self.position_codons,
            velocity=self.velocity_codons_per_sec,
            diameter=self.effective_diameter_codons,
            mass=self.effective_mass,
            metadata={
                'type': 'ribosome',
                'chain_length': self.nascent_chain.length_residues,
                'folding_state': self.nascent_chain.folding_state.value,
                'is_stalled': self.is_stalled
            }
        )
    
    def advance(self, dt: float):
        """
        Advance ribosome by dt seconds.
        
        This handles:
        1. Position update (from velocity)
        2. Nascent chain growth (1 codon = 1 amino acid)
        3. Time tracking
        """
        if not self.is_stalled:
            # Update position
            codons_translated = self.velocity_codons_per_sec * dt
            self.position_codons += codons_translated
            
            # Grow nascent chain (1 codon = 1 amino acid in standard genetic code)
            new_residues = int(codons_translated)
            if new_residues > 0:
                self.nascent_chain.grow(new_residues)
        
        self.time_since_initiation += dt


# =============================================================================
# Circular mRNA Model
# =============================================================================

@dataclass
class CircularMRNA:
    """
    Model of a circularized mRNA.
    
    Eukaryotic mRNAs are often functionally circular due to
    eIF4E-eIF4G-PABP interactions bridging the 5' cap and 3' poly(A) tail.
    
    This enables Closed-Loop-Assisted Reinitiation (CLAR) where
    terminating ribosomes can rapidly reinitiate at the start codon.
    
    Anatomy (in 5' to 3' direction):
    - 5' UTR (untranslated region)
    - Start codon (AUG)
    - Coding sequence (CDS)
    - Stop codon
    - 3' UTR
    - [circular link back to 5' end]
    """
    
    # Sequence lengths
    utr_5_length_codons: int = 50  # Before start codon
    cds_length_codons: int = 300   # Coding sequence
    utr_3_length_codons: int = 100  # After stop codon
    
    # Sequence-specific features (could affect local speed)
    codon_speeds: Optional[np.ndarray] = None  # Speed at each codon
    stall_sites: List[int] = field(default_factory=list)  # Known pause sites
    
    # Secondary structure effects
    structure_barriers: Dict[int, float] = field(default_factory=dict)
    # {position: energy_barrier} for RNA secondary structure
    
    @property
    def total_length_codons(self) -> int:
        """Total mRNA length in codon equivalents."""
        return self.utr_5_length_codons + self.cds_length_codons + self.utr_3_length_codons
    
    @property
    def start_codon_position(self) -> int:
        """Position of start codon (first CDS codon)."""
        return self.utr_5_length_codons
    
    @property
    def stop_codon_position(self) -> int:
        """Position of stop codon."""
        return self.utr_5_length_codons + self.cds_length_codons
    
    def get_local_speed_modifier(self, position: float) -> float:
        """
        Get speed modifier at a given position.
        
        Returns 1.0 for normal speed, <1.0 for slow regions.
        
        Factors that slow translation:
        - Rare codons (low tRNA availability)
        - RNA secondary structure
        - Specific stall sequences
        """
        if self.codon_speeds is None:
            return 1.0
        
        idx = int(position) % len(self.codon_speeds)
        return self.codon_speeds[idx]
    
    def is_in_cds(self, position: float) -> bool:
        """Check if position is within coding sequence."""
        return self.start_codon_position <= position < self.stop_codon_position
    
    @classmethod
    def create_typical_mrna(
        cls,
        protein_length_aa: int = 300,
        utr_5_length: int = 50,
        utr_3_length: int = 100,
        add_speed_variation: bool = True,
        seed: Optional[int] = None
    ) -> 'CircularMRNA':
        """
        Create an mRNA with typical properties.
        
        Args:
            protein_length_aa: Length of encoded protein
            utr_5_length: 5' UTR length in codon equivalents
            utr_3_length: 3' UTR length in codon equivalents
            add_speed_variation: Add codon-specific speed variation
            seed: Random seed
        """
        if seed is not None:
            np.random.seed(seed)
        
        mrna = cls(
            utr_5_length_codons=utr_5_length,
            cds_length_codons=protein_length_aa,
            utr_3_length_codons=utr_3_length
        )
        
        if add_speed_variation:
            # Create codon speed array with some slow spots
            total_length = mrna.total_length_codons
            speeds = np.ones(total_length)
            
            # Add some slow codons (rare codon clusters)
            n_slow_regions = max(1, protein_length_aa // 50)
            for _ in range(n_slow_regions):
                start = np.random.randint(utr_5_length, utr_5_length + protein_length_aa - 10)
                length = np.random.randint(3, 8)
                speeds[start:start+length] *= np.random.uniform(0.3, 0.7)
            
            mrna.codon_speeds = speeds
        
        return mrna


# =============================================================================
# Polyribosome System
# =============================================================================

class PolyribosomeSystem:
    """
    A complete polyribosome complex on circular mRNA.
    
    This is the main class for Paper 2. It wraps the core ParticleSystem
    with biological semantics and dynamics:
    
    1. Ribosomes initiate at the start codon
    2. They translate along the CDS, growing nascent chains
    3. At the stop codon, they terminate and can reinitiate (CLAR)
    4. Collisions between ribosomes create traffic jams
    
    The key insight: Because nascent chains grow at different rates
    (due to stalling, speed variation), the mass and diameter 
    distributions become heterogeneous → chaos.
    """
    
    def __init__(
        self,
        mrna: CircularMRNA,
        initiation_rate: float = 0.1,  # ribosomes/second
        base_elongation_rate: float = 5.0,  # codons/second
        enable_clar: bool = True,  # Closed-loop reinitiation
    ):
        self.mrna = mrna
        self.initiation_rate = initiation_rate
        self.base_elongation_rate = base_elongation_rate
        self.enable_clar = enable_clar
        
        # Create underlying particle system
        self.track = CircularTrack(mrna.total_length_codons)
        self.particle_system = ParticleSystem(mrna.total_length_codons)
        
        # Ribosome registry
        self.ribosomes: Dict[int, Ribosome] = {}
        self._next_id = 0
        
        # Statistics
        self.completed_translations = 0
        self.total_collisions = 0
        self.time = 0.0
    
    def initiate_ribosome(self) -> Optional[Ribosome]:
        """
        Try to initiate a new ribosome at the start codon.
        
        Fails if the initiation site is blocked by another ribosome.
        """
        start_pos = self.mrna.start_codon_position
        
        # Check if initiation site is clear
        for ribo in self.ribosomes.values():
            distance = self.track.distance(ribo.position_codons, start_pos)
            min_distance = (ribo.effective_diameter_codons + BIO_CONSTANTS.ribosome_footprint_codons) / 2
            if distance < min_distance:
                return None  # Blocked
        
        # Create new ribosome
        ribo = Ribosome(
            id=self._next_id,
            position_codons=float(start_pos),
            velocity_codons_per_sec=self.base_elongation_rate,
            nascent_chain=NascentChain(
                total_length=self.mrna.cds_length_codons
            )
        )
        
        self.ribosomes[ribo.id] = ribo
        
        # Add to particle system
        self.particle_system.add_particle(
            position=ribo.position_codons,
            velocity=ribo.velocity_codons_per_sec,
            diameter=ribo.effective_diameter_codons,
            mass=ribo.effective_mass,
            metadata={'ribosome_id': ribo.id}
        )
        
        self._next_id += 1
        return ribo
    
    def _sync_particles_from_ribosomes(self):
        """Update particle system to reflect current ribosome states."""
        for particle in self.particle_system.particles:
            ribo_id = particle.metadata.get('ribosome_id')
            if ribo_id is not None and ribo_id in self.ribosomes:
                ribo = self.ribosomes[ribo_id]
                particle.diameter = ribo.effective_diameter_codons
                particle.mass = ribo.effective_mass
    
    def _handle_termination(self, ribo: Ribosome):
        """Handle ribosome reaching stop codon."""
        self.completed_translations += 1
        
        if self.enable_clar:
            # Reinitiate: reset to start codon with new nascent chain
            ribo.position_codons = float(self.mrna.start_codon_position)
            ribo.nascent_chain = NascentChain(
                total_length=self.mrna.cds_length_codons
            )
            ribo.time_since_initiation = 0.0
        else:
            # Remove ribosome
            del self.ribosomes[ribo.id]
            # Remove from particle system
            self.particle_system.particles = [
                p for p in self.particle_system.particles
                if p.metadata.get('ribosome_id') != ribo.id
            ]
    
    def evolve(
        self,
        duration: float,
        dt: float = 0.01,
        record_density: bool = True,
        density_interval: float = 1.0
    ) -> Dict:
        """
        Evolve the polyribosome system.
        
        Returns dictionary with:
        - 'states': List of particle system states
        - 'densities': List of density fields (if recorded)
        - 'ribosome_data': Time series of ribosome properties
        - 'statistics': Summary statistics
        """
        results = {
            'states': [],
            'densities': [],
            'ribosome_data': [],
            'collision_events': [],
            'time_points': []
        }
        
        target_time = self.time + duration
        next_density_time = self.time
        
        while self.time < target_time:
            # Stochastic initiation
            if np.random.random() < self.initiation_rate * dt:
                self.initiate_ribosome()
            
            # Update ribosome states (grow nascent chains)
            for ribo in list(self.ribosomes.values()):
                ribo.advance(dt)
                
                # Check for termination
                if ribo.position_codons >= self.mrna.stop_codon_position:
                    self._handle_termination(ribo)
            
            # Sync changes to particle system
            self._sync_particles_from_ribosomes()
            
            # Evolve physics (collisions)
            states, collisions = self.particle_system.evolve(
                duration=dt,
                save_interval=dt
            )
            
            self.total_collisions += len(collisions)
            results['collision_events'].extend(collisions)
            
            # Record density field
            if record_density and self.time >= next_density_time:
                density = self.particle_system.compute_density_field(
                    n_bins=self.mrna.total_length_codons
                )
                results['densities'].append(density)
                results['time_points'].append(self.time)
                
                # Record ribosome data
                ribo_snapshot = {
                    'time': self.time,
                    'n_ribosomes': len(self.ribosomes),
                    'positions': [r.position_codons for r in self.ribosomes.values()],
                    'chain_lengths': [r.nascent_chain.length_residues for r in self.ribosomes.values()],
                    'masses': [r.effective_mass for r in self.ribosomes.values()],
                    'diameters': [r.effective_diameter_codons for r in self.ribosomes.values()]
                }
                results['ribosome_data'].append(ribo_snapshot)
                
                next_density_time += density_interval
            
            self.time += dt
        
        # Summary statistics
        results['statistics'] = {
            'duration': duration,
            'completed_translations': self.completed_translations,
            'total_collisions': self.total_collisions,
            'final_ribosome_count': len(self.ribosomes),
            'collision_rate': self.total_collisions / duration if duration > 0 else 0
        }
        
        return results
    
    def get_ribosome_density_profile(self) -> np.ndarray:
        """
        Get current density profile (compare to Ribo-seq data).
        
        This is what experimentalists measure!
        """
        return self.particle_system.compute_density_field(
            n_bins=self.mrna.total_length_codons
        )
    
    def get_mass_heterogeneity(self) -> float:
        """
        Compute coefficient of variation of masses.
        
        Higher = more heterogeneous = more chaotic.
        """
        masses = [r.effective_mass for r in self.ribosomes.values()]
        if len(masses) < 2:
            return 0.0
        return np.std(masses) / np.mean(masses)


# =============================================================================
# Comparison with TASEP
# =============================================================================

def tasep_steady_state_density(
    length: int,
    entry_rate: float,
    exit_rate: float,
    hopping_rate: float = 1.0
) -> np.ndarray:
    """
    Analytical TASEP steady-state density profile.
    
    This is what the standard model predicts.
    Our Hamiltonian model should give DIFFERENT predictions,
    especially for fluctuations and correlations.
    
    Uses mean-field approximation (exact for certain parameter regimes).
    """
    alpha = entry_rate / hopping_rate  # Normalized entry rate
    beta = exit_rate / hopping_rate    # Normalized exit rate
    
    density = np.zeros(length)
    
    if alpha < 0.5 and alpha < beta:
        # Low density phase
        rho = alpha
        density[:] = rho
    elif beta < 0.5 and beta < alpha:
        # High density phase
        rho = 1 - beta
        density[:] = rho
    else:
        # Maximum current phase
        rho = 0.5
        density[:] = rho
    
    return density


def compare_hamiltonian_vs_tasep(
    polysome_system: PolyribosomeSystem,
    tasep_entry_rate: float,
    tasep_exit_rate: float,
    duration: float = 1000.0
) -> Dict:
    """
    Compare our Hamiltonian model predictions with TASEP.
    
    Key differences to look for:
    1. Fluctuation magnitude (Hamiltonian may have larger fluctuations)
    2. Correlation structure (deterministic chaos vs Markovian)
    3. Traffic jam dynamics (Hamiltonian has inertia, TASEP doesn't)
    """
    # Run Hamiltonian simulation
    results = polysome_system.evolve(duration, record_density=True)
    
    # Get time-averaged density
    if results['densities']:
        hamiltonian_density = np.mean(results['densities'], axis=0)
        hamiltonian_std = np.std(results['densities'], axis=0)
    else:
        hamiltonian_density = np.zeros(polysome_system.mrna.total_length_codons)
        hamiltonian_std = np.zeros_like(hamiltonian_density)
    
    # Get TASEP prediction
    tasep_density = tasep_steady_state_density(
        length=polysome_system.mrna.total_length_codons,
        entry_rate=tasep_entry_rate,
        exit_rate=tasep_exit_rate
    )
    
    # Compute difference metrics
    mse = np.mean((hamiltonian_density - tasep_density) ** 2)
    correlation = np.corrcoef(hamiltonian_density, tasep_density)[0, 1]
    
    return {
        'hamiltonian_density': hamiltonian_density,
        'hamiltonian_fluctuations': hamiltonian_std,
        'tasep_density': tasep_density,
        'mse': mse,
        'correlation': correlation,
        'statistics': results['statistics']
    }
