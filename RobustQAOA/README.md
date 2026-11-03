# Memory-Constrained Adversarially Robust Neural Network Pruning Using QAOA

## Project Overview

This project investigates whether QAOA (Quantum Approximate Optimization Algorithm) can be used to select neural network channels under a memory constraint while incorporating adversarial robustness into the channel-selection objective.

The primary research question is:
**Does QAOA-based channel selection produce meaningfully different results from classical greedy pruning under the same experimental conditions?**

No a-priori assumption of quantum advantage is made. Results are measured and reported honestly.

---

## Method Summary

1. **Model**: SmallCNN with a 16-channel prunable bottleneck (`conv1`)
2. **Dataset**: CIFAR-100
3. **Channel selection as QUBO**:

   ```
   maximize: α Σ c_i x_i  +  β Σ r_i x_i  +  γ Σ Q_ij x_i x_j  -  λ(Σ w_i x_i - M)²
   ```

4. **Three solvers**: Classical greedy | Exact exhaustive | QAOA (Qiskit Aer)
5. **Iterative loop**: Defender selects channels → Attacker generates adversarial examples → Robustness scores updated → Repeat

---

## Project Structure

```
RobustQAOA/
├── configs/          debug.yaml, research.yaml
├── models/           cnn.py, pruned_model.py, model_utils.py
├── training/         train.py, evaluate.py
├── pruning/          importance.py, synergy.py, classical_pruning.py, pruning_utils.py
├── optimization/     qubo.py, exact_solver.py, qaoa.py
├── attacks/          fgsm.py, pgd.py
├── robustness/       attack_pool.py, robustness_score.py
├── experiments/      baseline.py, classical_experiment.py, exact_qubo_experiment.py,
│                     qaoa_experiment.py, comparison.py, defender_attacker_loop.py,
│                     ablations.py, report_generator.py
├── verification/     bounded_verification.py
├── visualization/    plots.py
├── tests/            test_model.py, test_pruning.py, test_qubo.py,
│                     test_attacks.py, test_qaoa.py
├── results/          (generated at runtime)
├── main.py
└── requirements.txt
```

---

## Installation

```bash
pip install -r requirements.txt
```

Required libraries: `torch`, `torchvision`, `qiskit`, `qiskit-aer`, `qiskit-algorithms`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `PyYAML`, `pytest`, `tqdm`, `scipy`

---

## Usage

### Debug mode (fast, for development only)
```bash
python main.py --config configs/debug.yaml
```

### Research / full mode (final results)
```bash
python main.py --config configs/research.yaml
```

### Force retraining
```bash
python main.py --config configs/research.yaml --force-retrain
```

### Run tests first
```bash
pytest tests/ -v
```

---

## Output

All results are saved to `results/`:

| File | Description |
|------|-------------|
| `results/checkpoints/baseline_model.pt` | Best trained model checkpoint |
| `results/baseline/` | Baseline training metrics and curves |
| `results/classical/` | Classical pruning results |
| `results/exact_qubo/` | Exact QUBO solver results |
| `results/qaoa/` | QAOA results |
| `results/comparison/comparison.md` | Main comparison table (markdown) |
| `results/comparison/comparison.csv` | Main comparison table (CSV) |
| `results/iterations/iterations.csv` | Defender-attacker loop per-round results |
| `results/final_report.md` | Full research report |
| `results/environment.json` | Python/PyTorch/Qiskit version info |

---

## Key Design Decisions

### Why 16 channels?
With n=16 binary variables, the exact solver needs to check 2^16 = 65,536 bitstrings, and QAOA can be demonstrated on a circuit with 16 qubits — tractable on a simulator.

### Memory cost formula
Each channel i costs 319 parameters × 4 bytes = 1,276 bytes:
- conv1 filter: 3×3×3 = 27 params
- bn1 parameters: weight + bias + running_mean + running_var = 4 params
- conv2 input slice: 32 filters × 3×3 = 288 params

### Sign convention
The QUBO objective is a **maximization** problem. The QAOA cost Hamiltonian **minimizes** the negated objective. All reported objective values are in maximization form (higher = better).

### Scientific integrity
- Results come from actual execution — no hard-coded or fabricated values
- If QAOA performs similarly to or worse than classical, that is reported honestly
- The exact QUBO result serves as the ground-truth optimum

---

## Adversarial Attacks

- **FGSM**: `x_adv = x + ε · sign(∇_x L)`
- **PGD**: Iterative projected gradient descent with random start

Both attacks use real model gradients. No simulated or random attacks.

---

## Reproducibility

Set `seed: 42` in config (default). All NumPy, PyTorch, and Python random states are seeded. Environment info is saved to `results/environment.json`.

---

## Limitations

- CIFAR-100 benchmark only; generalization to other domains not evaluated
- QAOA uses a simulator (Qiskit Aer) — not real quantum hardware
- Robustness is empirical (FGSM/PGD); no formal certification
- Exhaustive input-space verification is infeasible for CIFAR-100

---

## Citation

If you use this code for research, please cite appropriately and note that QAOA was run via Qiskit Aer simulation, not on real quantum hardware.
