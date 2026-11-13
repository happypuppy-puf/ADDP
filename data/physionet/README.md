# PhysioNet 2012 data

Run the following command from the repository root:

```bash
python scripts/download_data.py --dataset physionet
```

The downloader obtains the official 4,000-record `set-a` archive used by the
paper. The training script reads the archive directly and builds its local
patient cache on the first run. `set-b` and `set-c` are not merged into this
cohort.
