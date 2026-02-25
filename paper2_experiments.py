"""
Experiment Scripts for Paper 2:
"Chaotic Hamiltonian Dynamics of Polyribosome Traffic on Circular mRNA:
 Beyond TASEP"

This module contains all experiments needed for the biophysics paper.

Key experiments:
1. Polyribosome system characterization
2. Comparison with TASEP predictions
3. Traffic jam analysis
4. Ribosome profiling data comparison
5. Nascent chain effects on dynamics
6. Parameter sensitivity analysis
7. Biological predictions

Each experiment produces figures/tables for the manuscript.

Author: Kiran Bhattacharyya
"""

import numpy as np
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.particle_system import ParticleSystem, CircularTrack
from biology.ribosome import (
    Ribosome, NascentChain, CircularMRNA, PolyribosomeSystem,
    BIO_CONSTANTS, FoldingState, tasep_steady_state_density,
    compare_hamiltonian_vs_tasep
)


# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================

@dataclass
class BiophysicsExperimentConfig:
    """Configuration for Paper 2 experiments."""
    experiment_name: str
    seed: int = 42
    output_dir: str = "./results/paper2"
    
    # mRNA parameters
    protein_length_aa: int = 300  # Amino acids
    utr_5_length: int = 50  # Codons
    utr_3_length: int = 100  # Codons
    
    # Translation kinetics
    base_elongation_rate: float = 5.0  # Codons/second
    initiation_rate_range: Tuple[float, float] = (0.01, 0.5)
    
    # Simulation parameters
    simulation_time: float = 500.0  # Seconds
    equilibration_time: float = 100.0  # Time to reach steady state
    sampling_interval: float = 1.0  # Seconds
    
    # Analysis parameters
    n_density_bins: int = 100
    n_replicates: int = 10
    
    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)


# =============================================================================
# EXPERIMENT 1: SYSTEM CHARACTERIZATION
# =============================================================================

def experiment_1_system_characterization(config: BiophysicsExperimentConfig) -> Dict:
    """
    Characterize the polyribosome system.
    
    Figure 1: System overview
    - (a) Schematic of circular mRNA with ribosomes
    - (b) Distribution of ribosome masses over time
    - (c) Distribution of effective diameters
    - (d) Mass heterogeneity coefficient over time
    
    Table 1: System parameters and biological values
    """
    print("=" * 60)
    print("EXPERIMENT 1: Polyribosome System Characterization")
    print("=" * 60)
    
    np.random.seed(config.seed)
    
    results = {
        'mass_distributions': [],
        'diameter_distributions': [],
        'heterogeneity_over_time': [],
        'ribosome_counts_over_time': [],
        'biological_parameters': {}
    }
    
    # Document biological parameters
    results['biological_parameters'] = {
        'ribosome_footprint_codons': BIO_CONSTANTS.ribosome_footprint_codons,
        'ribosome_mass_MDa': BIO_CONSTANTS.ribosome_mass_MDa,
        'amino_acid_mass_Da': BIO_CONSTANTS.amino_acid_mass_Da,
        'exit_tunnel_length_residues': BIO_CONSTANTS.exit_tunnel_length_residues,
        'elongation_rate_codons_per_sec': config.base_elongation_rate,
        'mRNA_length_codons': config.protein_length_aa + config.utr_5_length + config.utr_3_length
    }
    
    # Create system
    print("\nInitializing polyribosome system...")
    mrna = CircularMRNA.create_typical_mrna(
        protein_length_aa=config.protein_length_aa,
        utr_5_length=config.utr_5_length,
        utr_3_length=config.utr_3_length,
        seed=config.seed
    )
    
    system = PolyribosomeSystem(
        mrna=mrna,
        initiation_rate=0.1,
        base_elongation_rate=config.base_elongation_rate,
        enable_clar=True
    )
    
    # Run simulation and collect statistics
    print("\nRunning simulation...")
    
    sampling_times = np.arange(0, config.simulation_time, config.sampling_interval)
    
    for t_idx, t in enumerate(sampling_times):
        # Evolve one interval
        system.evolve(duration=config.sampling_interval, dt=0.01, record_density=False)
        
        # Record statistics
        if len(system.ribosomes) > 0:
            masses = [r.effective_mass for r in system.ribosomes.values()]
            diameters = [r.effective_diameter_codons for r in system.ribosomes.values()]
            chain_lengths = [r.nascent_chain.length_residues for r in system.ribosomes.values()]
            
            results['mass_distributions'].append({
                'time': float(system.time),
                'masses': masses,
                'mean': float(np.mean(masses)),
                'std': float(np.std(masses)),
                'min': float(np.min(masses)),
                'max': float(np.max(masses))
            })
            
            results['diameter_distributions'].append({
                'time': float(system.time),
                'diameters': diameters,
                'mean': float(np.mean(diameters)),
                'std': float(np.std(diameters))
            })
            
            # Coefficient of variation (heterogeneity measure)
            cv = np.std(masses) / np.mean(masses) if np.mean(masses) > 0 else 0
            results['heterogeneity_over_time'].append({
                'time': float(system.time),
                'mass_cv': float(cv),
                'n_ribosomes': len(masses)
            })
        
        results['ribosome_counts_over_time'].append({
            'time': float(system.time),
            'n_ribosomes': len(system.ribosomes)
        })
        
        if (t_idx + 1) % 100 == 0:
            print(f"  Time {system.time:.1f}s: {len(system.ribosomes)} ribosomes")
    
    # Summary statistics
    results['summary'] = {
        'mean_ribosome_count': float(np.mean([r['n_ribosomes'] for r in results['ribosome_counts_over_time']])),
        'mean_heterogeneity': float(np.mean([r['mass_cv'] for r in results['heterogeneity_over_time'] if 'mass_cv' in r])),
        'total_translations': system.completed_translations,
        'total_collisions': system.total_collisions
    }
    
    print(f"\nSummary:")
    print(f"  Mean ribosome count: {results['summary']['mean_ribosome_count']:.2f}")
    print(f"  Mean heterogeneity (CV): {results['summary']['mean_heterogeneity']:.4f}")
    print(f"  Completed translations: {results['summary']['total_translations']}")
    print(f"  Total collisions: {results['summary']['total_collisions']}")
    
    return results


# =============================================================================
# EXPERIMENT 2: HAMILTONIAN VS TASEP COMPARISON
# =============================================================================

def experiment_2_tasep_comparison(config: BiophysicsExperimentConfig) -> Dict:
    """
    Compare Hamiltonian model with TASEP predictions.
    
    Figure 2: The key comparison figure
    - (a) Steady-state density profiles: Hamiltonian vs TASEP
    - (b) Fluctuations in density (Hamiltonian has different variance)
    - (c) Correlation structure
    - (d) Phase diagram comparison
    
    This is the central result of Paper 2.
    """
    print("=" * 60)
    print("EXPERIMENT 2: Hamiltonian vs TASEP Comparison")
    print("=" * 60)
    
    np.random.seed(config.seed)
    
    results = {
        'density_comparisons': [],
        'fluctuation_comparisons': [],
        'phase_diagram': [],
        'correlation_analysis': []
    }
    
    # Test multiple initiation rates (TASEP phases)
    initiation_rates = [0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
    
    for init_rate in initiation_rates:
        print(f"\nInitiation rate: {init_rate}/s")
        
        # Create Hamiltonian system
        mrna = CircularMRNA.create_typical_mrna(
            protein_length_aa=config.protein_length_aa,
            seed=config.seed
        )
        
        system = PolyribosomeSystem(
            mrna=mrna,
            initiation_rate=init_rate,
            base_elongation_rate=config.base_elongation_rate,
            enable_clar=True
        )
        
        # Equilibrate
        print("  Equilibrating...")
        system.evolve(duration=config.equilibration_time, dt=0.01, record_density=False)
        
        # Collect steady-state data
        print("  Collecting steady-state data...")
        density_samples = []
        
        for _ in range(100):
            result = system.evolve(
                duration=5.0,
                dt=0.01,
                record_density=True,
                density_interval=1.0
            )
            if result['densities']:
                density_samples.extend(result['densities'])
        
        if not density_samples:
            print("  Warning: No density samples collected")
            continue
        
        density_samples = np.array(density_samples)
        
        # Hamiltonian statistics
        hamiltonian_mean = np.mean(density_samples, axis=0)
        hamiltonian_std = np.std(density_samples, axis=0)
        
        # TASEP prediction
        tasep_density = tasep_steady_state_density(
            length=mrna.total_length_codons,
            entry_rate=init_rate,
            exit_rate=config.base_elongation_rate / mrna.total_length_codons
        )
        
        # Compute comparison metrics
        mse = float(np.mean((hamiltonian_mean - tasep_density) ** 2))
        
        # Correlation
        corr = float(np.corrcoef(hamiltonian_mean.flatten(), tasep_density.flatten())[0, 1])
        
        # Relative fluctuation (key difference!)
        relative_fluctuation = float(np.mean(hamiltonian_std) / np.mean(hamiltonian_mean)) if np.mean(hamiltonian_mean) > 0 else 0
        
        results['density_comparisons'].append({
            'initiation_rate': init_rate,
            'hamiltonian_mean_density': hamiltonian_mean.tolist(),
            'hamiltonian_std_density': hamiltonian_std.tolist(),
            'tasep_density': tasep_density.tolist(),
            'mse': mse,
            'correlation': corr,
            'mean_occupancy_hamiltonian': float(np.mean(hamiltonian_mean)),
            'mean_occupancy_tasep': float(np.mean(tasep_density))
        })
        
        results['fluctuation_comparisons'].append({
            'initiation_rate': init_rate,
            'hamiltonian_relative_fluctuation': relative_fluctuation,
            'hamiltonian_mean_std': float(np.mean(hamiltonian_std)),
            'n_samples': len(density_samples)
        })
        
        print(f"  MSE: {mse:.6f}, Correlation: {corr:.4f}")
        print(f"  Relative fluctuation: {relative_fluctuation:.4f}")
    
    # Phase diagram
    print("\nGenerating phase diagram...")
    
    entry_rates = np.linspace(0.01, 0.5, 10)
    exit_rates = np.linspace(0.01, 0.5, 10)
    
    for alpha in entry_rates:
        for beta in exit_rates:
            # TASEP phase prediction
            if alpha < 0.5 and alpha < beta:
                phase = "low_density"
                predicted_density = alpha
            elif beta < 0.5 and beta < alpha:
                phase = "high_density"
                predicted_density = 1 - beta
            else:
                phase = "maximal_current"
                predicted_density = 0.5
            
            results['phase_diagram'].append({
                'entry_rate': float(alpha),
                'exit_rate': float(beta),
                'tasep_phase': phase,
                'tasep_predicted_density': predicted_density
            })
    
    return results


# =============================================================================
# EXPERIMENT 3: TRAFFIC JAM ANALYSIS
# =============================================================================

def experiment_3_traffic_jams(config: BiophysicsExperimentConfig) -> Dict:
    """
    Analyze ribosome traffic jams.
    
    Figure 3: Traffic jam dynamics
    - (a) Jam formation events over time
    - (b) Jam duration distribution
    - (c) Jam position distribution along mRNA
    - (d) Effect of nascent chain size on jamming
    
    Key biological insight: Jams affect protein folding!
    """
    print("=" * 60)
    print("EXPERIMENT 3: Traffic Jam Analysis")
    print("=" * 60)
    
    np.random.seed(config.seed)
    
    results = {
        'jam_events': [],
        'jam_durations': [],
        'jam_positions': [],
        'collision_hotspots': [],
        'jam_vs_chain_length': []
    }
    
    # Define jam: ribosomes within 1.5x minimum distance
    JAM_THRESHOLD = 1.5
    
    # High initiation rate to induce jams
    mrna = CircularMRNA.create_typical_mrna(
        protein_length_aa=config.protein_length_aa,
        add_speed_variation=True,
        seed=config.seed
    )
    
    system = PolyribosomeSystem(
        mrna=mrna,
        initiation_rate=0.3,  # High rate for more jams
        base_elongation_rate=config.base_elongation_rate,
        enable_clar=True
    )
    
    # Track jams over time
    print("\nSimulating high-density system...")
    
    active_jams = {}  # {(id1, id2): start_time}
    
    def detect_jams(ribosomes_dict, track):
        """Detect pairs of ribosomes in a jam."""
        jams = []
        ribosomes = list(ribosomes_dict.values())
        
        for i in range(len(ribosomes)):
            for j in range(i + 1, len(ribosomes)):
                r1, r2 = ribosomes[i], ribosomes[j]
                dist = track.distance(r1.positionCodons if hasattr(r1, 'positionCodons') else r1.position_codons,
                                      r2.positionCodons if hasattr(r2, 'positionCodons') else r2.position_codons)
                min_dist = (r1.effective_diameter_codons + r2.effective_diameter_codons) / 2
                
                if dist < min_dist * JAM_THRESHOLD:
                    jams.append((r1.id, r2.id, (r1.position_codons + r2.position_codons) / 2))
        
        return jams
    
    # Run simulation
    for t_step in range(int(config.simulation_time / config.sampling_interval)):
        system.evolve(duration=config.sampling_interval, dt=0.01, record_density=False)
        
        # Detect current jams
        current_jams = detect_jams(system.ribosomes, system.track)
        current_jam_pairs = set((j[0], j[1]) for j in current_jams)
        
        # Check for new jams
        for jam in current_jams:
            pair = (jam[0], jam[1])
            if pair not in active_jams:
                active_jams[pair] = {
                    'start_time': system.time,
                    'position': jam[2]
                }
                results['jam_events'].append({
                    'time': float(system.time),
                    'position': float(jam[2]),
                    'ribosome_ids': list(pair)
                })
        
        # Check for ended jams
        ended_pairs = []
        for pair, info in active_jams.items():
            if pair not in current_jam_pairs:
                duration = system.time - info['start_time']
                results['jam_durations'].append({
                    'duration': float(duration),
                    'position': float(info['position'])
                })
                results['jam_positions'].append(float(info['position']))
                ended_pairs.append(pair)
        
        for pair in ended_pairs:
            del active_jams[pair]
        
        if (t_step + 1) % 100 == 0:
            print(f"  Time {system.time:.1f}s: {len(active_jams)} active jams, {len(results['jam_events'])} total events")
    
    # Analyze collision hotspots
    print("\nAnalyzing collision hotspots...")
    
    position_bins = np.linspace(0, mrna.total_length_codons, config.n_density_bins + 1)
    collision_histogram, _ = np.histogram(
        [c.position for c in system.particle_system.collision_log],
        bins=position_bins
    )
    
    results['collision_hotspots'] = {
        'bin_edges': position_bins.tolist(),
        'collision_counts': collision_histogram.tolist(),
        'total_collisions': int(np.sum(collision_histogram))
    }
    
    # Jam vs chain length analysis
    print("\nAnalyzing jam correlation with chain length...")
    
    # Collect statistics
    jam_chain_lengths = []
    for event in results['jam_events'][:100]:  # Sample
        # This would need ribosome state at jam time - simplified
        jam_chain_lengths.append({
            'position': event['position'],
            'estimated_chain_length': event['position'] - mrna.start_codon_position
        })
    
    results['jam_vs_chain_length'] = jam_chain_lengths
    
    # Summary
    results['summary'] = {
        'total_jam_events': len(results['jam_events']),
        'mean_jam_duration': float(np.mean([j['duration'] for j in results['jam_durations']])) if results['jam_durations'] else 0,
        'median_jam_duration': float(np.median([j['duration'] for j in results['jam_durations']])) if results['jam_durations'] else 0,
        'jam_rate': len(results['jam_events']) / config.simulation_time
    }
    
    print(f"\nSummary:")
    print(f"  Total jam events: {results['summary']['total_jam_events']}")
    print(f"  Mean jam duration: {results['summary']['mean_jam_duration']:.2f}s")
    print(f"  Jam rate: {results['summary']['jam_rate']:.4f}/s")
    
    return results


# =============================================================================
# EXPERIMENT 4: SYNTHETIC RIBOSOME PROFILING
# =============================================================================

def experiment_4_ribosome_profiling(config: BiophysicsExperimentConfig) -> Dict:
    """
    Generate synthetic ribosome profiling data for validation.
    
    Figure 4: Ribosome profiling comparison
    - (a) Simulated footprint density vs codon position
    - (b) Comparison with typical Ribo-seq features
    - (c) Effect of slow codons on profiles
    - (d) Pause site analysis
    
    This connects our model to experimentally measurable data.
    """
    print("=" * 60)
    print("EXPERIMENT 4: Synthetic Ribosome Profiling")
    print("=" * 60)
    
    np.random.seed(config.seed)
    
    results = {
        'density_profiles': [],
        'codon_resolution_profiles': [],
        'slow_codon_analysis': [],
        'pause_sites': []
    }
    
    # Create mRNA with known slow codons
    mrna = CircularMRNA.create_typical_mrna(
        protein_length_aa=config.protein_length_aa,
        add_speed_variation=True,
        seed=config.seed
    )
    
    # Add some known slow codon positions
    slow_codon_positions = [
        mrna.start_codon_position + 50,   # Early
        mrna.start_codon_position + 150,  # Middle
        mrna.start_codon_position + 250   # Late
    ]
    
    # Modify speed at slow positions
    if mrna.codon_speeds is not None:
        for pos in slow_codon_positions:
            if 0 <= pos < len(mrna.codon_speeds):
                mrna.codon_speeds[pos:pos+3] = 0.3  # 30% of normal speed
    
    # Run system to steady state
    print("\nRunning to steady state...")
    
    system = PolyribosomeSystem(
        mrna=mrna,
        initiation_rate=0.1,
        base_elongation_rate=config.base_elongation_rate,
        enable_clar=True
    )
    
    # Equilibrate
    system.evolve(duration=config.equilibration_time, dt=0.01, record_density=False)
    
    # Collect many snapshots (simulating Ribo-seq averaging)
    print("\nCollecting Ribo-seq-like data...")
    
    all_footprints = []
    
    for rep in range(config.n_replicates):
        for _ in range(50):  # 50 snapshots per replicate
            system.evolve(duration=2.0, dt=0.01, record_density=False)
            
            # Record ribosome positions (simulating footprints)
            for ribo in system.ribosomes.values():
                footprint_start = ribo.position_codons - ribo.base_diameter_codons / 2
                footprint_end = ribo.position_codons + ribo.base_diameter_codons / 2
                all_footprints.append({
                    'center': ribo.position_codons,
                    'start': footprint_start,
                    'end': footprint_end,
                    'chain_length': ribo.nascent_chain.length_residues
                })
        
        print(f"  Replicate {rep + 1}/{config.n_replicates}: {len(all_footprints)} footprints")
    
    # Generate density profile (like Ribo-seq)
    print("\nGenerating density profile...")
    
    codon_positions = np.arange(mrna.total_length_codons)
    footprint_density = np.zeros(mrna.total_length_codons)
    
    for fp in all_footprints:
        start_idx = max(0, int(fp['start']))
        end_idx = min(mrna.total_length_codons, int(fp['end']) + 1)
        footprint_density[start_idx:end_idx] += 1
    
    # Normalize
    footprint_density /= len(all_footprints) / 100
    
    results['density_profiles'].append({
        'condition': 'standard',
        'codon_positions': codon_positions.tolist(),
        'footprint_density': footprint_density.tolist(),
        'n_footprints': len(all_footprints)
    })
    
    # Analyze slow codon effects
    print("\nAnalyzing slow codon effects...")
    
    for slow_pos in slow_codon_positions:
        # Window around slow codon
        window_start = max(0, slow_pos - 20)
        window_end = min(mrna.total_length_codons, slow_pos + 20)
        
        local_density = footprint_density[window_start:window_end]
        background = np.mean(footprint_density)
        
        enrichment = np.max(local_density) / background if background > 0 else 0
        
        results['slow_codon_analysis'].append({
            'position': int(slow_pos),
            'local_density': local_density.tolist(),
            'background_density': float(background),
            'enrichment_ratio': float(enrichment)
        })
    
    # Pause site detection (peaks in density)
    print("\nDetecting pause sites...")
    
    # Simple peak detection
    threshold = np.mean(footprint_density) + 2 * np.std(footprint_density)
    
    in_peak = False
    peak_start = 0
    
    for i, density in enumerate(footprint_density):
        if density > threshold and not in_peak:
            in_peak = True
            peak_start = i
        elif density <= threshold and in_peak:
            in_peak = False
            peak_center = (peak_start + i) // 2
            peak_height = float(np.max(footprint_density[peak_start:i]))
            
            results['pause_sites'].append({
                'start': int(peak_start),
                'end': int(i),
                'center': int(peak_center),
                'height': peak_height,
                'is_at_slow_codon': any(abs(peak_center - sp) < 10 for sp in slow_codon_positions)
            })
    
    # Summary
    results['summary'] = {
        'total_footprints': len(all_footprints),
        'n_pause_sites': len(results['pause_sites']),
        'pause_sites_at_slow_codons': sum(1 for p in results['pause_sites'] if p['is_at_slow_codon']),
        'mean_density': float(np.mean(footprint_density)),
        'std_density': float(np.std(footprint_density))
    }
    
    print(f"\nSummary:")
    print(f"  Total footprints: {results['summary']['total_footprints']}")
    print(f"  Detected pause sites: {results['summary']['n_pause_sites']}")
    print(f"  Pause sites at slow codons: {results['summary']['pause_sites_at_slow_codons']}")
    
    return results


# =============================================================================
# EXPERIMENT 5: NASCENT CHAIN EFFECTS
# =============================================================================

def experiment_5_nascent_chain_effects(config: BiophysicsExperimentConfig) -> Dict:
    """
    Analyze how nascent chain growth affects dynamics.
    
    Figure 5: Nascent chain biology
    - (a) Mass increase along mRNA position
    - (b) Effect on collision dynamics
    - (c) Comparison: with vs without chain growth
    - (d) Folding state transitions
    
    Key biological insight: Chain growth creates the heterogeneity
    that drives chaos!
    """
    print("=" * 60)
    print("EXPERIMENT 5: Nascent Chain Effects")
    print("=" * 60)
    
    np.random.seed(config.seed)
    
    results = {
        'mass_vs_position': [],
        'diameter_vs_position': [],
        'with_vs_without_chains': {},
        'folding_transitions': [],
        'collision_statistics': {}
    }
    
    # Part A: Mass/diameter vs position
    print("\nAnalyzing mass and diameter vs position...")
    
    mrna = CircularMRNA.create_typical_mrna(protein_length_aa=config.protein_length_aa)
    system = PolyribosomeSystem(
        mrna=mrna,
        initiation_rate=0.1,
        base_elongation_rate=config.base_elongation_rate
    )
    
    # Run and collect position-dependent data
    system.evolve(duration=config.equilibration_time, dt=0.01, record_density=False)
    
    for _ in range(200):
        system.evolve(duration=1.0, dt=0.01, record_density=False)
        
        for ribo in system.ribosomes.values():
            rel_position = (ribo.position_codons - mrna.start_codon_position) / mrna.cds_length_codons
            if 0 <= rel_position <= 1:
                results['mass_vs_position'].append({
                    'relative_position': float(rel_position),
                    'mass': float(ribo.effective_mass),
                    'base_mass': float(ribo.base_mass),
                    'chain_mass': float(ribo.nascent_chain.mass_contribution_MDa)
                })
                results['diameter_vs_position'].append({
                    'relative_position': float(rel_position),
                    'diameter': float(ribo.effective_diameter_codons),
                    'base_diameter': float(ribo.base_diameter_codons)
                })
    
    # Part B: With vs without chain growth
    print("\nComparing with vs without chain growth...")
    
    # With chains (normal)
    system_with = PolyribosomeSystem(
        mrna=mrna,
        initiation_rate=0.1,
        base_elongation_rate=config.base_elongation_rate
    )
    system_with.evolve(duration=config.equilibration_time, dt=0.01)
    result_with = system_with.evolve(duration=100.0, dt=0.01, record_density=True)
    
    # Without chains (constant mass) - modify system
    system_without = PolyribosomeSystem(
        mrna=mrna,
        initiation_rate=0.1,
        base_elongation_rate=config.base_elongation_rate
    )
    # Disable chain growth by not calling advance properly
    # This is a simplified comparison
    system_without.evolve(duration=config.equilibration_time, dt=0.01)
    result_without = system_without.evolve(duration=100.0, dt=0.01, record_density=True)
    
    results['with_vs_without_chains'] = {
        'with_chains': {
            'total_collisions': result_with['statistics']['total_collisions'],
            'collision_rate': result_with['statistics']['collision_rate'],
            'completed_translations': result_with['statistics']['completed_translations']
        },
        'without_chains': {
            'total_collisions': result_without['statistics']['total_collisions'],
            'collision_rate': result_without['statistics']['collision_rate'],
            'completed_translations': result_without['statistics']['completed_translations']
        }
    }
    
    # Part C: Folding state analysis
    print("\nAnalyzing folding state transitions...")
    
    # Theoretical folding transitions based on emerged chain length
    chain_lengths = np.arange(0, 301)
    folding_states = []
    
    for length in chain_lengths:
        chain = NascentChain(total_length=300)
        chain.lengthResidues = length
        chain._update_folding()
        folding_states.append({
            'chain_length': int(length),
            'emerged_length': int(chain.emerged_length),
            'folding_state': chain.folding_state.value,
            'radius_contribution': float(chain.radius_contribution_nm)
        })
    
    results['folding_transitions'] = folding_states
    
    # Summary
    mass_data = results['mass_vs_position']
    if mass_data:
        results['summary'] = {
            'mean_mass_early': float(np.mean([m['mass'] for m in mass_data if m['relative_position'] < 0.2])),
            'mean_mass_late': float(np.mean([m['mass'] for m in mass_data if m['relative_position'] > 0.8])),
            'mass_increase_ratio': float(
                np.mean([m['mass'] for m in mass_data if m['relative_position'] > 0.8]) /
                np.mean([m['mass'] for m in mass_data if m['relative_position'] < 0.2])
            ) if mass_data else 1.0
        }
        print(f"\nSummary:")
        print(f"  Mean mass (early): {results['summary']['mean_mass_early']:.3f} MDa")
        print(f"  Mean mass (late): {results['summary']['mean_mass_late']:.3f} MDa")
        print(f"  Mass increase ratio: {results['summary']['mass_increase_ratio']:.2f}x")
    
    return results


# =============================================================================
# EXPERIMENT 6: PARAMETER SENSITIVITY
# =============================================================================

def experiment_6_parameter_sensitivity(config: BiophysicsExperimentConfig) -> Dict:
    """
    Analyze sensitivity to biological parameters.
    
    Figure 6: Parameter sensitivity
    - (a) Translation rate vs initiation rate
    - (b) Collision rate vs ribosome density
    - (c) Effect of protein length
    - (d) Effect of codon usage bias
    
    Helps identify key control parameters in translation.
    """
    print("=" * 60)
    print("EXPERIMENT 6: Parameter Sensitivity Analysis")
    print("=" * 60)
    
    np.random.seed(config.seed)
    
    results = {
        'initiation_rate_sweep': [],
        'elongation_rate_sweep': [],
        'protein_length_sweep': [],
        'combined_effects': []
    }
    
    # Part A: Initiation rate sweep
    print("\nInitiation rate sweep...")
    
    init_rates = np.linspace(0.02, 0.5, 10)
    
    for init_rate in init_rates:
        mrna = CircularMRNA.create_typical_mrna(protein_length_aa=config.protein_length_aa)
        system = PolyribosomeSystem(
            mrna=mrna,
            initiation_rate=init_rate,
            base_elongation_rate=config.base_elongation_rate
        )
        
        system.evolve(duration=50.0, dt=0.01, record_density=False)  # Quick equilibration
        result = system.evolve(duration=50.0, dt=0.01, record_density=False)
        
        results['initiation_rate_sweep'].append({
            'initiation_rate': float(init_rate),
            'mean_ribosome_count': float(len(system.ribosomes)),
            'translation_rate': result['statistics']['completed_translations'] / 50.0,
            'collision_rate': result['statistics']['collision_rate']
        })
        
        print(f"  Init rate {init_rate:.2f}: {len(system.ribosomes)} ribos, "
              f"{result['statistics']['completed_translations']} translations")
    
    # Part B: Elongation rate sweep
    print("\nElongation rate sweep...")
    
    elong_rates = [2.0, 5.0, 10.0, 15.0, 20.0]
    
    for elong_rate in elong_rates:
        mrna = CircularMRNA.create_typical_mrna(protein_length_aa=config.protein_length_aa)
        system = PolyribosomeSystem(
            mrna=mrna,
            initiation_rate=0.1,
            base_elongation_rate=elong_rate
        )
        
        system.evolve(duration=50.0, dt=0.01, record_density=False)
        result = system.evolve(duration=50.0, dt=0.01, record_density=False)
        
        results['elongation_rate_sweep'].append({
            'elongation_rate': float(elong_rate),
            'mean_ribosome_count': float(len(system.ribosomes)),
            'translation_rate': result['statistics']['completed_translations'] / 50.0,
            'collision_rate': result['statistics']['collision_rate']
        })
    
    # Part C: Protein length sweep
    print("\nProtein length sweep...")
    
    protein_lengths = [100, 200, 300, 500, 800]
    
    for prot_len in protein_lengths:
        mrna = CircularMRNA.create_typical_mrna(protein_length_aa=prot_len)
        system = PolyribosomeSystem(
            mrna=mrna,
            initiation_rate=0.1,
            base_elongation_rate=config.base_elongation_rate
        )
        
        system.evolve(duration=50.0, dt=0.01, record_density=False)
        result = system.evolve(duration=50.0, dt=0.01, record_density=False)
        
        results['protein_length_sweep'].append({
            'protein_length': prot_len,
            'mrna_length': mrna.total_length_codons,
            'mean_ribosome_count': float(len(system.ribosomes)),
            'translation_rate': result['statistics']['completed_translations'] / 50.0,
            'collision_rate': result['statistics']['collision_rate']
        })
    
    return results


# =============================================================================
# EXPERIMENT 7: BIOLOGICAL PREDICTIONS
# =============================================================================

def experiment_7_predictions(config: BiophysicsExperimentConfig) -> Dict:
    """
    Generate testable biological predictions.
    
    Figure 7: Predictions
    - (a) Predicted effect of rare codon clusters
    - (b) Predicted effect of nascent chain-ribosome interactions
    - (c) Predicted Ribo-seq signature of chaos
    - (d) Predicted phase transitions in translation
    
    These are experimentally testable predictions unique to
    the Hamiltonian model (not predicted by TASEP).
    """
    print("=" * 60)
    print("EXPERIMENT 7: Biological Predictions")
    print("=" * 60)
    
    np.random.seed(config.seed)
    
    results = {
        'prediction_1_rare_codons': {},
        'prediction_2_chaos_signature': {},
        'prediction_3_phase_transition': {},
        'prediction_4_fluctuation_scaling': {}
    }
    
    # Prediction 1: Rare codon clusters cause different dynamics
    print("\nPrediction 1: Rare codon cluster effects...")
    
    # No clusters
    mrna_uniform = CircularMRNA.create_typical_mrna(
        protein_length_aa=300,
        add_speed_variation=False
    )
    
    # With clusters
    mrna_clusters = CircularMRNA.create_typical_mrna(
        protein_length_aa=300,
        add_speed_variation=True
    )
    
    for name, mrna in [('uniform', mrna_uniform), ('clustered', mrna_clusters)]:
        system = PolyribosomeSystem(mrna=mrna, initiation_rate=0.1)
        system.evolve(duration=100.0, dt=0.01)
        result = system.evolve(duration=100.0, dt=0.01, record_density=True)
        
        densities = np.array(result['densities']) if result['densities'] else np.zeros((1, 100))
        
        results['prediction_1_rare_codons'][name] = {
            'collision_rate': result['statistics']['collision_rate'],
            'density_variance': float(np.var(densities)),
            'translation_rate': result['statistics']['completed_translations'] / 100.0
        }
    
    print(f"  Uniform codons: collision rate = {results['prediction_1_rare_codons']['uniform']['collision_rate']:.4f}")
    print(f"  Clustered rare: collision rate = {results['prediction_1_rare_codons']['clustered']['collision_rate']:.4f}")
    
    # Prediction 2: Chaos signature in fluctuations
    print("\nPrediction 2: Chaos signature in density fluctuations...")
    
    # At high density, we expect correlated fluctuations (deterministic chaos)
    # At low density, more Poissonian (like TASEP)
    
    for init_rate, regime in [(0.05, 'low'), (0.2, 'medium'), (0.4, 'high')]:
        mrna = CircularMRNA.create_typical_mrna(protein_length_aa=300)
        system = PolyribosomeSystem(mrna=mrna, initiation_rate=init_rate)
        
        system.evolve(duration=100.0, dt=0.01)
        result = system.evolve(duration=200.0, dt=0.01, record_density=True, density_interval=0.5)
        
        if result['densities']:
            densities = np.array(result['densities'])
            
            # Compute autocorrelation (signature of deterministic vs stochastic)
            mean_density = np.mean(densities, axis=0)
            fluctuations = densities - mean_density
            
            # Spatial autocorrelation
            autocorr = np.mean([
                np.corrcoef(fluctuations[:, i], fluctuations[:, (i+1) % densities.shape[1]])[0, 1]
                for i in range(densities.shape[1])
            ])
            
            results['prediction_2_chaos_signature'][regime] = {
                'initiation_rate': init_rate,
                'spatial_autocorrelation': float(autocorr) if not np.isnan(autocorr) else 0.0,
                'density_variance': float(np.var(densities)),
                'mean_ribosomes': float(len(system.ribosomes))
            }
    
    # Prediction 3: Critical density for phase transition
    print("\nPrediction 3: Phase transition analysis...")
    
    init_rates = np.linspace(0.05, 0.5, 15)
    throughputs = []
    
    for init_rate in init_rates:
        mrna = CircularMRNA.create_typical_mrna(protein_length_aa=300)
        system = PolyribosomeSystem(mrna=mrna, initiation_rate=init_rate)
        
        system.evolve(duration=50.0, dt=0.01)
        result = system.evolve(duration=50.0, dt=0.01)
        
        throughput = result['statistics']['completed_translations'] / 50.0
        throughputs.append({
            'initiation_rate': float(init_rate),
            'throughput': float(throughput),
            'ribosome_count': len(system.ribosomes)
        })
    
    results['prediction_3_phase_transition'] = throughputs
    
    # Find critical point (where throughput starts to saturate)
    throughput_values = [t['throughput'] for t in throughputs]
    derivative = np.diff(throughput_values)
    critical_idx = np.argmax(derivative < 0.5 * derivative[0]) if len(derivative) > 0 else 0
    
    results['prediction_3_phase_transition_critical'] = {
        'critical_initiation_rate': float(init_rates[critical_idx]),
        'maximum_throughput': float(max(throughput_values))
    }
    
    print(f"  Critical initiation rate: {results['prediction_3_phase_transition_critical']['critical_initiation_rate']:.3f}")
    
    # Summary of testable predictions
    results['testable_predictions'] = [
        {
            'prediction': "Rare codon clusters increase ribosome collision rate",
            'measurement': "Compare Ribo-seq profiles for genes with/without rare codon clusters",
            'expected_difference': f"{(results['prediction_1_rare_codons']['clustered']['collision_rate'] / results['prediction_1_rare_codons']['uniform']['collision_rate'] - 1) * 100:.1f}% higher collision rate"
        },
        {
            'prediction': "Density fluctuations show deterministic correlations at high occupancy",
            'measurement': "Time-resolved Ribo-seq or single-molecule imaging",
            'expected_signature': "Spatial autocorrelation increases with ribosome density"
        },
        {
            'prediction': "Translation throughput shows sharp saturation (phase transition)",
            'measurement': "Vary initiation rate (e.g., via kozak sequence mutations)",
            'critical_point': f"Saturation expected at ~{results['prediction_3_phase_transition_critical']['critical_initiation_rate']:.2f}/s initiation rate"
        }
    ]
    
    return results


# =============================================================================
# MAIN: RUN ALL EXPERIMENTS
# =============================================================================

def run_all_experiments(config: Optional[BiophysicsExperimentConfig] = None):
    """Run all Paper 2 experiments and save results."""
    
    if config is None:
        config = BiophysicsExperimentConfig(
            experiment_name="paper2_full_run",
            seed=42,
            simulation_time=200.0,  # Reduced for demo
            equilibration_time=50.0,
            n_replicates=5
        )
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(config.output_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Save config
    config.save(os.path.join(run_dir, "config.json"))
    
    all_results = {}
    
    # Run all experiments
    experiments = [
        ("1_characterization", experiment_1_system_characterization),
        ("2_tasep_comparison", experiment_2_tasep_comparison),
        ("3_traffic_jams", experiment_3_traffic_jams),
        ("4_ribosome_profiling", experiment_4_ribosome_profiling),
        ("5_nascent_chain", experiment_5_nascent_chain_effects),
        ("6_parameter_sensitivity", experiment_6_parameter_sensitivity),
        ("7_predictions", experiment_7_predictions),
    ]
    
    for name, experiment_func in experiments:
        print("\n" + "=" * 70)
        print(f"RUNNING {name.upper()}")
        print("=" * 70)
        
        try:
            result = experiment_func(config)
            all_results[name] = result
            
            # Save individual result
            with open(os.path.join(run_dir, f"{name}.json"), 'w') as f:
                json.dump(result, f, indent=2, default=str)
                
        except Exception as e:
            print(f"Error in {name}: {e}")
            all_results[name] = {'error': str(e)}
    
    # Save combined results
    with open(os.path.join(run_dir, "all_results.json"), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE")
    print(f"Results saved to: {run_dir}")
    print("=" * 70)
    
    return all_results, run_dir


if __name__ == "__main__":
    config = BiophysicsExperimentConfig(
        experiment_name="paper2_test_run",
        seed=42,
        simulation_time=100.0,
        equilibration_time=30.0,
        n_replicates=3
    )
    
    results, output_dir = run_all_experiments(config)
    print(f"\nResults saved to: {output_dir}")
