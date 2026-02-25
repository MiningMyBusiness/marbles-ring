# Polyribosome Dynamics: Experiment Scripts

This directory contains all experiment scripts needed to generate results for two papers:

## Paper 1: Neural Operators for Impulsive Dynamical Systems
**Target venues:** NeurIPS, ICML, Journal of Computational Physics

### Experiments
| Experiment | Description | Key Figures |
|------------|-------------|-------------|
| 1. Dataset Generation | Generate diverse 1D hard-sphere trajectories | Fig S1: Dataset statistics |
| 2. Lyapunov Characterization | Quantify chaos in the system | Fig 2: Lyapunov analysis |
| 3. FNO Training | Train Fourier Neural Operator | Fig 3: Training curves |
| 4. Lyapunov Bypass | **Key result**: Predict beyond Lyapunov time | Fig 4: MSE vs horizon |
| 5. Ablation Studies | Architecture sensitivity | Table 2: Ablations |
| 6. Baseline Comparisons | Compare with other methods | Fig 6: Method comparison |
| 7. Inverse Problem | Infer masses from collisions | Fig 7: Mass inference |

### Running Paper 1 Experiments
```bash
cd experiments
python -m paper1.run_experiments
```

---

## Paper 2: Chaotic Hamiltonian Dynamics of Polyribosome Traffic
**Target venues:** Biophysical Journal, PLOS Computational Biology

### Experiments
| Experiment | Description | Key Figures |
|------------|-------------|-------------|
| 1. System Characterization | Document polyribosome properties | Fig 1: System overview |
| 2. TASEP Comparison | **Key result**: Compare with standard model | Fig 2: Hamiltonian vs TASEP |
| 3. Traffic Jam Analysis | Analyze ribosome collisions | Fig 3: Jam dynamics |
| 4. Ribosome Profiling | Synthetic Ribo-seq data | Fig 4: Density profiles |
| 5. Nascent Chain Effects | How chains drive heterogeneity | Fig 5: Chain growth |
| 6. Parameter Sensitivity | Biological parameter space | Fig 6: Sensitivity |
| 7. Biological Predictions | Testable predictions | Fig 7: Predictions |

### Running Paper 2 Experiments
```bash
cd experiments
python -m paper2.run_experiments
```

---

## Running All Experiments

### Full run (takes ~1-2 hours)
```bash
python run_all.py
```

### Quick test run (~5-10 minutes)
```bash
python run_all.py --quick
```

### Run specific paper
```bash
python run_all.py --paper 1  # Paper 1 only
python run_all.py --paper 2  # Paper 2 only
```

### Generate figures from existing results
```bash
python run_all.py --figures-only
```

---

## Output Structure

```
results/
├── paper1/
│   └── run_YYYYMMDD_HHMMSS/
│       ├── config.json
│       ├── dataset.json
│       ├── results.json
│       ├── figures/
│       │   ├── fig2_lyapunov.png
│       │   ├── fig4_bypass.png
│       │   └── ...
│       └── figure_data_csv/
│           └── ...
│
└── paper2/
    └── run_YYYYMMDD_HHMMSS/
        ├── config.json
        ├── 1_characterization.json
        ├── 2_tasep_comparison.json
        ├── ...
        ├── all_results.json
        ├── figures/
        └── figure_data_csv/
```

---

## Key Results to Highlight

### Paper 1
1. **Lyapunov bypass**: FNO predicts density evolution up to 5-10× Lyapunov time
2. **Discontinuity handling**: First neural operator for impulsive (non-smooth) dynamics
3. **Inverse problem**: Can infer particle masses from collision timestamps alone

### Paper 2
1. **TASEP comparison**: Hamiltonian model predicts different fluctuation statistics
2. **Traffic jams**: Deterministic chaos creates correlated jam dynamics
3. **Nascent chain heterogeneity**: Growing chains are the key driver of chaos
4. **Testable predictions**: Specific predictions for Ribo-seq experiments

---

## Requirements

```
numpy>=1.20
matplotlib>=3.4 (optional, for figure generation)
```

For GPU-accelerated training (Paper 1):
```
torch>=1.10
```

---

## Citation

If you use this code, please cite both papers:

```bibtex
@article{paper1,
  title={Neural Operators for Density Evolution in Impulsive Dynamical Systems},
  author={...},
  journal={...},
  year={2024}
}

@article{paper2,
  title={Chaotic Hamiltonian Dynamics of Polyribosome Traffic on Circular mRNA: Beyond TASEP},
  author={...},
  journal={...},
  year={2024}
}
```
