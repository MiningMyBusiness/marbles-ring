#!/usr/bin/env python3
"""
Main Runner for Polyribosome Dynamics Experiments

This script runs all experiments for both papers:
- Paper 1: Neural Operators for Impulsive Dynamical Systems
- Paper 2: Chaotic Hamiltonian Dynamics of Polyribosome Traffic

Usage:
    python run_all.py                    # Run both papers
    python run_all.py --paper 1          # Run Paper 1 only
    python run_all.py --paper 2          # Run Paper 2 only
    python run_all.py --quick            # Quick test run
    python run_all.py --figures-only     # Generate figures from existing results

Author: Kiran Bhattacharyya
"""

import argparse
import os
import sys
import json
from datetime import datetime

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def run_paper1_experiments(quick: bool = False) -> str:
    """Run all Paper 1 experiments."""
    from experiments.paper1.run_experiments import (
        run_all_experiments, ExperimentConfig
    )
    
    if quick:
        config = ExperimentConfig(
            experiment_name="paper1_quick_test",
            seed=42,
            n_training_trajectories=20,
            n_validation_trajectories=5,
            n_test_trajectories=5,
            simulation_time=20.0
        )
    else:
        config = ExperimentConfig(
            experiment_name="paper1_full",
            seed=42,
            n_training_trajectories=500,
            n_validation_trajectories=50,
            n_test_trajectories=50,
            simulation_time=100.0
        )
    
    results, output_dir = run_all_experiments(config)
    return output_dir


def run_paper2_experiments(quick: bool = False) -> str:
    """Run all Paper 2 experiments."""
    from experiments.paper2.run_experiments import (
        run_all_experiments, BiophysicsExperimentConfig
    )
    
    if quick:
        config = BiophysicsExperimentConfig(
            experiment_name="paper2_quick_test",
            seed=42,
            simulation_time=50.0,
            equilibration_time=20.0,
            n_replicates=2
        )
    else:
        config = BiophysicsExperimentConfig(
            experiment_name="paper2_full",
            seed=42,
            simulation_time=500.0,
            equilibration_time=100.0,
            n_replicates=10
        )
    
    results, output_dir = run_all_experiments(config)
    return output_dir


def generate_figures(paper1_dir: str = None, paper2_dir: str = None):
    """Generate figures from experiment results."""
    from experiments.figure_generation import (
        prepare_paper1_figure_data,
        prepare_paper2_figure_data,
        export_figure_data_to_csv,
        generate_figure_summary,
        plot_paper1_figures,
        plot_paper2_figures
    )
    
    # Paper 1 figures
    if paper1_dir and os.path.exists(paper1_dir):
        results_path = os.path.join(paper1_dir, "results.json")
        if os.path.exists(results_path):
            print(f"\nGenerating Paper 1 figures from {paper1_dir}")
            with open(results_path, 'r') as f:
                results = json.load(f)
            
            figures_dir = os.path.join(paper1_dir, "figures")
            plot_paper1_figures(results, figures_dir)
            
            # Also export CSV
            figure_data = prepare_paper1_figure_data(results)
            csv_dir = os.path.join(paper1_dir, "figure_data_csv")
            export_figure_data_to_csv(figure_data, csv_dir)
            
            # Summary
            summary = generate_figure_summary(figure_data)
            with open(os.path.join(paper1_dir, "figure_summary.txt"), 'w') as f:
                f.write(summary)
    
    # Paper 2 figures
    if paper2_dir and os.path.exists(paper2_dir):
        results_path = os.path.join(paper2_dir, "all_results.json")
        if os.path.exists(results_path):
            print(f"\nGenerating Paper 2 figures from {paper2_dir}")
            with open(results_path, 'r') as f:
                results = json.load(f)
            
            figures_dir = os.path.join(paper2_dir, "figures")
            plot_paper2_figures(results, figures_dir)
            
            # Also export CSV
            figure_data = prepare_paper2_figure_data(results)
            csv_dir = os.path.join(paper2_dir, "figure_data_csv")
            export_figure_data_to_csv(figure_data, csv_dir)


def print_summary(paper1_dir: str = None, paper2_dir: str = None):
    """Print summary of completed experiments."""
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    
    if paper1_dir:
        print(f"\nPaper 1 results: {paper1_dir}")
        results_path = os.path.join(paper1_dir, "results.json")
        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                results = json.load(f)
            
            print("  Experiments completed:")
            for key in results.keys():
                print(f"    - {key}")
    
    if paper2_dir:
        print(f"\nPaper 2 results: {paper2_dir}")
        results_path = os.path.join(paper2_dir, "all_results.json")
        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                results = json.load(f)
            
            print("  Experiments completed:")
            for key in results.keys():
                if 'error' not in results[key]:
                    print(f"    - {key} ✓")
                else:
                    print(f"    - {key} ✗ (error)")
    
    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Run experiments for polyribosome dynamics papers"
    )
    parser.add_argument(
        "--paper", type=int, choices=[1, 2],
        help="Run experiments for specific paper (1 or 2). Default: both"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick test run with reduced parameters"
    )
    parser.add_argument(
        "--figures-only", action="store_true",
        help="Only generate figures from existing results"
    )
    parser.add_argument(
        "--paper1-dir", type=str,
        help="Path to Paper 1 results directory (for --figures-only)"
    )
    parser.add_argument(
        "--paper2-dir", type=str,
        help="Path to Paper 2 results directory (for --figures-only)"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("POLYRIBOSOME DYNAMICS EXPERIMENT RUNNER")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mode: {'Quick test' if args.quick else 'Full run'}")
    
    paper1_dir = args.paper1_dir
    paper2_dir = args.paper2_dir
    
    if args.figures_only:
        if not paper1_dir and not paper2_dir:
            # Find most recent results
            p1_base = "./results/paper1"
            p2_base = "./results/paper2"
            
            if os.path.exists(p1_base):
                runs = sorted([d for d in os.listdir(p1_base) if d.startswith("run_")])
                if runs:
                    paper1_dir = os.path.join(p1_base, runs[-1])
            
            if os.path.exists(p2_base):
                runs = sorted([d for d in os.listdir(p2_base) if d.startswith("run_")])
                if runs:
                    paper2_dir = os.path.join(p2_base, runs[-1])
        
        generate_figures(paper1_dir, paper2_dir)
        print_summary(paper1_dir, paper2_dir)
        return
    
    # Run experiments
    if args.paper is None or args.paper == 1:
        print("\n" + "=" * 70)
        print("RUNNING PAPER 1 EXPERIMENTS")
        print("=" * 70)
        paper1_dir = run_paper1_experiments(quick=args.quick)
    
    if args.paper is None or args.paper == 2:
        print("\n" + "=" * 70)
        print("RUNNING PAPER 2 EXPERIMENTS")
        print("=" * 70)
        paper2_dir = run_paper2_experiments(quick=args.quick)
    
    # Generate figures
    generate_figures(paper1_dir, paper2_dir)
    
    # Print summary
    print_summary(paper1_dir, paper2_dir)
    
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
