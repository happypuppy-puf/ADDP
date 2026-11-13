# Data preparation

Run all commands from the repository root. Dataset splits are performed over
Activity recording IDs, patients, or stations before sample construction.
Training-set historical observations are used for normalization.

## Activity

Place `ConfLongDemo_JSI.txt` at:

```text
data/activity/ConfLongDemo_JSI.txt
```

Download it with:

```bash
python scripts/download_data.py --dataset activity
```

The downloader obtains the same raw file used by
[t-PatchGNN](https://github.com/usail-hkust/t-PatchGNN/tree/main/data/activity)
and verifies its SHA-256 checksum.

## PhysioNet

Download the PhysioNet 2012 challenge record archives with:

```bash
python scripts/download_data.py --dataset physionet
```

The official 4,000-record `set-a` archive is placed under:

```text
data/physionet/
```

The first run creates `patient_data_cache.pkl` locally. The generated cache is
excluded from version control. `set-b` and `set-c` are not merged into the
paper's benchmark cohort.

## USHCN

Place `small_chunked_sporadic.csv` at:

```text
data/ushcn/small_chunked_sporadic.csv
```

Download it with:

```bash
python scripts/download_data.py --dataset ushcn
```

The downloader obtains the same raw file used by
[t-PatchGNN](https://github.com/usail-hkust/t-PatchGNN/tree/main/data/ushcn)
and verifies its SHA-256 checksum.

## MIMIC-III

Follow the authorized-access instructions in [`mimic/README.md`](mimic/README.md).
