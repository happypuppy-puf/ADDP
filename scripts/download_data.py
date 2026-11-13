#!/usr/bin/env python3
"""Download public benchmark files used by ADDP."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil
import urllib.error
import urllib.request


ROOT = Path(__file__).resolve().parents[1]

PUBLIC_FILES = {
    "activity": {
        "path": ROOT / "data" / "activity" / "ConfLongDemo_JSI.txt",
        "sha256": "1fbc965b8d7faf499ea854428b2138132d94e6a26404bc8e4b0c62144f513e8a",
        "urls": [
            "https://archive.ics.uci.edu/ml/machine-learning-databases/00196/ConfLongDemo_JSI.txt",
            "https://cdn.jsdelivr.net/gh/usail-hkust/t-PatchGNN@main/data/activity/raw/ConfLongDemo_JSI.txt",
        ],
    },
    "ushcn": {
        "path": ROOT / "data" / "ushcn" / "small_chunked_sporadic.csv",
        "sha256": "671eb8d121522e98891c84197742a6c9e9bb5015e42b328a93ebdf2cfd393ecf",
        "urls": [
            "https://cdn.jsdelivr.net/gh/usail-hkust/t-PatchGNN@main/data/ushcn/raw/small_chunked_sporadic.csv",
            "https://raw.githubusercontent.com/usail-hkust/t-PatchGNN/main/data/ushcn/raw/small_chunked_sporadic.csv",
        ],
    },
}

PHYSIONET_URLS = [
    "https://physionet.org/files/challenge-2012/1.0.0/set-a.tar.gz?download",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stream_download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "ADDP-data-downloader"})
    with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output, length=1024 * 1024)
    temporary.replace(destination)


def download_verified(name: str) -> None:
    specification = PUBLIC_FILES[name]
    destination = specification["path"]
    expected = specification["sha256"]
    if destination.exists() and sha256(destination) == expected:
        print(f"{name}: already available at {destination}")
        return

    errors = []
    for url in specification["urls"]:
        try:
            print(f"Downloading {name} from {url}")
            stream_download(url, destination)
            actual = sha256(destination)
            if actual != expected:
                raise RuntimeError(
                    f"checksum mismatch: expected {expected}, received {actual}"
                )
            print(f"Saved {destination}")
            return
        except (OSError, RuntimeError, urllib.error.URLError) as error:
            errors.append(f"{url}: {error}")
    raise RuntimeError("\n".join(errors))


def download_physionet() -> None:
    directory = ROOT / "data" / "physionet"
    directory.mkdir(parents=True, exist_ok=True)
    for url in PHYSIONET_URLS:
        filename = url.split("/")[-1].split("?")[0]
        destination = directory / filename
        if destination.exists() and destination.stat().st_size > 0:
            print(f"physionet: already available at {destination}")
            continue
        print(f"Downloading PhysioNet archive from {url}")
        stream_download(url, destination)
        print(f"Saved {destination}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["activity", "ushcn", "physionet", "all"],
    )
    args = parser.parse_args()

    selected = ["activity", "ushcn", "physionet"] if args.dataset == "all" else [args.dataset]
    for dataset in selected:
        if dataset == "physionet":
            download_physionet()
        else:
            download_verified(dataset)


if __name__ == "__main__":
    main()
