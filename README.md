# ADDP

Official implementation of **Adaptive Data-Dependent Pruning for Irregular
Multivariate Time-Series Forecasting**.

ADDP computes a sample-level roughness index and uses it as a shared routing
signal for three coordinated decisions during training:

1. data-dependent patch scoring;
2. shared-core retention across a conservative and an aggressive view;
3. temperature-weighted smooth worst-view optimization.

The two views share one forecasting backbone. Inference uses the complete
history in a single forward pass.

## Repository structure

```text
ADDP/
|-- addp/
|   `-- routing.py
|-- data/
|   |-- activity/
|   |-- mimic/
|   |-- physionet/
|   `-- ushcn/
|-- scripts/
|   |-- run_all.sh
|   |-- train_activity.py
|   |-- train_mimic.py
|   |-- train_physionet.py
|   `-- train_ushcn.py
|-- run_models.py
`-- requirements.txt
```

## Environment

Python 3.10 or later and PyTorch 2.1 or later are recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate`.

## Data

Download the public datasets with:

```bash
python scripts/download_data.py --dataset activity
python scripts/download_data.py --dataset ushcn
python scripts/download_data.py --dataset physionet
```

See [`data/README.md`](data/README.md) for sources and expected file locations.
Activity and USHCN use the same raw files as the t-PatchGNN benchmark.
MIMIC-III follows its credentialed-access requirements.

Activity recording IDs are divided into training, validation, and test sets
before window generation. Each set then uses 4-second windows with a
500-millisecond step, so overlapping windows from the same recording cannot
cross dataset partitions. All normalization statistics are computed from the
training partition only.

## Training

Run one dataset from the repository root:

```bash
python run_models.py --dataset activity --seed 0
python run_models.py --dataset physionet --seed 0
python run_models.py --dataset ushcn --seed 0
python run_models.py --dataset mimic --seed 0
```

Run all four experiments:

```bash
bash scripts/run_all.sh
```

Repeat a command with seeds `0` through `4` to reproduce the five-seed
reporting protocol. Each checkpoint stores both the forecasting backbone and
the learned calibration parameters.

The default training protocol uses Adam, gradient clipping at 5, at most 50
epochs, and checkpoint selection by validation MSE. Evaluation always uses the
complete, unpruned history.

| Dataset | Forecasting task | Batch size |
|---|---|---:|
| Activity | 3-second history to the next second | 32 |
| PhysioNet | first 24 hours to the following 24 hours | 32 |
| USHCN | 24 months to the following month | 64 |
| MIMIC-III | first 24 hours to the following 24 hours | 8 |

## Learnable routing

The forward pass uses hard sorting, shared-core construction, and integer
retention. The complete learned shared core is retained in both views. Hard
ranking, rounding, and top-k memberships are fixed during backpropagation.
The score calibrator is updated by a continuous kept-versus-dropped margin
loss, the shared-core calibrator by a continuous core-ratio regularizer, and
the temperature calibrator directly by the smooth worst-view forecasting
objective. ADDP does not use REINFORCE or a straight-through estimator.

The implementation is centralized in [`addp/routing.py`](addp/routing.py) and
is shared by all dataset scripts.

The released configuration uses a score margin of `0.1`, a shared-core target
of `0.45`, and a core-regularization weight of `1e-4`. The score-loss weight is
`1e-2` for Activity and `1e-3` for PhysioNet, USHCN, and MIMIC-III. The fixed
calibration bounds are `rho in [0.2, 0.7]` and `tau in [0.3, 2.5]`.

## Dataset distribution

Raw datasets and generated caches are intentionally not tracked in Git. The
public Activity, PhysioNet, and USHCN inputs are obtained by the downloader;
MIMIC-III remains subject to PhysioNet credentialed access. This keeps the
repository small and avoids redistributing data under a different license.

## Acknowledgements

The forecasting backbone and benchmark preparation follow
[t-PatchGNN](https://github.com/usail-hkust/t-PatchGNN).
