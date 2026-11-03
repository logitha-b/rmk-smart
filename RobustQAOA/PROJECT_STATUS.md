# PROJECT STATUS

## Status: READY TO RUN

All source files have been created. The project is ready for execution.

---

## File Checklist

### Core
- [x] `main.py` — main orchestrator
- [x] `requirements.txt`
- [x] `README.md`
- [x] `.gitignore`

### Configuration
- [x] `configs/debug.yaml`
- [x] `configs/research.yaml`

### Models
- [x] `models/cnn.py` — SmallCNN with 16-channel prunable conv1
- [x] `models/pruned_model.py` — Physical weight-transfer pruning
- [x] `models/model_utils.py` — Parameter count, memory, inference time

### Training
- [x] `training/train.py` — Full training loop + CIFAR-100 loading
- [x] `training/evaluate.py` — Evaluation with sklearn metrics

### Pruning
- [x] `pruning/importance.py` — Channel ablation importance
- [x] `pruning/synergy.py` — Pairwise synergy matrix
- [x] `pruning/classical_pruning.py` — Greedy importance-based selection
- [x] `pruning/pruning_utils.py` — Memory costs, feasibility

### Optimization
- [x] `optimization/qubo.py` — QUBO formulation + Ising conversion
- [x] `optimization/exact_solver.py` — Exhaustive 2^16 enumeration
- [x] `optimization/qaoa.py` — QAOA via Qiskit Aer + classical fallback

### Attacks
- [x] `attacks/fgsm.py` — FGSM with gradient
- [x] `attacks/pgd.py` — PGD with random start

### Robustness
- [x] `robustness/attack_pool.py` — Attack pool construction
- [x] `robustness/robustness_score.py` — Per-channel r_i scores

### Experiments
- [x] `experiments/baseline.py`
- [x] `experiments/classical_experiment.py`
- [x] `experiments/exact_qubo_experiment.py`
- [x] `experiments/qaoa_experiment.py`
- [x] `experiments/comparison.py`
- [x] `experiments/defender_attacker_loop.py`
- [x] `experiments/ablations.py`
- [x] `experiments/report_generator.py`

### Other
- [x] `verification/bounded_verification.py`
- [x] `visualization/plots.py`

### Tests
- [x] `tests/test_model.py`
- [x] `tests/test_pruning.py`
- [x] `tests/test_qubo.py`
- [x] `tests/test_attacks.py`
- [x] `tests/test_qaoa.py`

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Debug run (fast)
python main.py --config configs/debug.yaml

# Full research run
python main.py --config configs/research.yaml
```

---

## Notes

- QAOA uses Qiskit Aer (statevector simulator). If Qiskit is unavailable, a classical Monte Carlo fallback is used and labeled clearly as `qaoa_fallback`.
- The project automatically downloads CIFAR-100 via torchvision.
- All results are saved to `results/`. The final comparison is in `results/comparison/comparison.md`.
- The final research report is at `results/final_report.md`.
