# Activity data

Run the following command from the repository root:

```bash
python scripts/download_data.py --dataset activity
```

The downloader saves `ConfLongDemo_JSI.txt` in this directory and verifies its
SHA-256 checksum.

`recording_split.json` contains the exact 15/5/5 recording-ID split used by
the revised leakage-controlled experiment. The split is applied before
overlapping 4-second windows are generated.
