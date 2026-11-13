# USHCN data

Run the following command from the repository root:

```bash
python scripts/download_data.py --dataset ushcn
```

The downloader saves `small_chunked_sporadic.csv` in this directory and
verifies its SHA-256 checksum.
