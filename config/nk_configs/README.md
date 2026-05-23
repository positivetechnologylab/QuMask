# Per-(n,k) Configuration Files

This directory contains one YAML config per (n,k) experimental configuration. Each file is a copy of `../default.yaml` with four fields overridden:

| Field | Value |
|-------|-------|
| `data.n` | Total system qubits |
| `data.k` | Target qubits |
| `ensemble.M` | 3 (reduced from 10 for computational budget) |
| `paths.data_dir` | `data/processed/n{n}_k{k}` |
| `paths.checkpoint_dir` | `checkpoints/n{n}_k{k}` |
| `paths.results_dir` | `results/n{n}_k{k}` |

All other fields (dataset sizes, seeds, depth range, model hyperparameters, training hyperparameters) are identical across all configs.

## Configurations

| Config | Data source |
|--------|-------------|
| `n6_k3.yaml` – `n14_k3.yaml` | Derived from `n15_k3` master via `data/derive_nk.py` (no re-simulation needed) |
| `n15_k3.yaml` | Fresh generation via `train.py --generate-data` |
| `n15_k4.yaml` | Fresh generation |
| `n15_k5.yaml` | Fresh generation |
| `n15_k6.yaml` | Fresh generation |

## Usage

```bash
# Generate master (15,3) dataset (covers all n',3 derivations)
python3 train.py --config config/nk_configs/n15_k3.yaml --generate-data

# Derive (n',3) datasets for n'=6..14
for n in 6 7 8 9 10 11 12 13 14; do
    python3 data/derive_nk.py \
        --master-dir data/processed/n15_k3 \
        --n-prime $n \
        --out-dir data/processed/n${n}_k3
done

# Generate (15,4), (15,5), (15,6) datasets
python3 train.py --config config/nk_configs/n15_k4.yaml --generate-data
python3 train.py --config config/nk_configs/n15_k5.yaml --generate-data
python3 train.py --config config/nk_configs/n15_k6.yaml --generate-data

# Train one (n,k) on Modal
python3 modal_train.py upload --config config/nk_configs/n6_k3.yaml
python3 modal_train.py train  --config config/nk_configs/n6_k3.yaml
python3 modal_train.py download --config config/nk_configs/n6_k3.yaml
```
