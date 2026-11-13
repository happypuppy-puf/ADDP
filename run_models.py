#!/usr/bin/env python3
"""Unified launcher for the four main experiments."""
import argparse
import os
import subprocess
import sys

PROGRAMS = {
    "activity": "train_activity.py",
    "physionet": "train_physionet.py",
    "ushcn": "train_ushcn.py",
    "mimic": "train_mimic.py",
}

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True, choices=PROGRAMS)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()
root = os.path.dirname(os.path.abspath(__file__))
environment = os.environ.copy()
environment["ADDP_SEED"] = str(args.seed)
raise SystemExit(subprocess.call(
    [sys.executable, os.path.join(root, "scripts", PROGRAMS[args.dataset])],
    cwd=root,
    env=environment,
))
