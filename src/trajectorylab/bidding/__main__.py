"""Run with python -m trajectorylab.bidding --config configs/bidding-pilot.toml."""

import argparse
from pathlib import Path

from .experiment import run

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", type=Path, required=True)
parser.add_argument("--output", type=Path)
parser.add_argument("--stage", choices=("all", "data", "train", "evaluate"), default="all")
args = parser.parse_args()
run(args.config, args.output, args.stage)
