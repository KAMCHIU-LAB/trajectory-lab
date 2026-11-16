"""Small command-line surface; all artifacts are local and credentials are unnecessary."""

from __future__ import annotations

import argparse
import functools
import http.server
import json
import os
from pathlib import Path

from trajectorylab.config import load_config
from trajectorylab.experiment import evaluate_checkpoint, prepare_data, run_suite
from trajectorylab.report import generate_report
from trajectorylab.utils import read_json, runtime_info


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        prog="trajectorylab", description="Auditable offline RL data experiments"
    )
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Print runtime and accelerator availability")
    for name in ("prepare", "run"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
    report = commands.add_parser("report")
    report.add_argument("--suite", type=Path)
    report.add_argument("--output", type=Path, default=Path("reports/latest"))
    checkpoint = commands.add_parser("evaluate")
    checkpoint.add_argument("--checkpoint", type=Path, required=True)
    checkpoint.add_argument("--seeds", type=int, nargs="+", required=True)
    server = commands.add_parser("serve")
    server.add_argument("--directory", type=Path, default=Path("reports/latest"))
    server.add_argument("--port", type=int, default=8765)
    return result


def main() -> None:
    args = parser().parse_args()
    root = Path.cwd()
    os.environ.setdefault("MINARI_DATASETS_PATH", str(root / "artifacts" / "minari"))
    if args.command == "doctor":
        print(json.dumps(runtime_info(), indent=2))
    elif args.command == "prepare":
        _, _, _, audit = prepare_data(load_config(args.config), root)
        print(json.dumps(audit, indent=2))
    elif args.command == "run":
        directory = run_suite(load_config(args.config), root)
        destination = root / "reports" / directory.name
        path = generate_report(directory, root, destination)
        print(f"Complete suite: {directory}\nReport: {path}")
    elif args.command == "report":
        directory = args.suite
        if directory is None:
            directory = root / read_json(root / "artifacts/latest.json")["suite_directory"]
        print(generate_report(directory, root, args.output))
    elif args.command == "evaluate":
        print(json.dumps(evaluate_checkpoint(args.checkpoint, args.seeds), indent=2))
    elif args.command == "serve":
        if not (args.directory / "index.html").is_file():
            raise FileNotFoundError(f"No report at {args.directory}; run an experiment first")
        handler = functools.partial(
            http.server.SimpleHTTPRequestHandler, directory=str(args.directory.resolve())
        )
        server = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), handler)
        print(f"Report available at http://127.0.0.1:{args.port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
