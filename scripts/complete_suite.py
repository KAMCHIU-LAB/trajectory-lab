"""Resume an initialized suite with independent CPU workers.

Stop the original runner before using this command. This driver preserves every
run identity and training setting, records its own source digest, and delegates
all learning/checkpoint behavior to the same reference experiment implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from trajectorylab.data import Normalizer
from trajectorylab.evaluation import evaluate
from trajectorylab.experiment import implementation_digest, prepare_data, train_run
from trajectorylab.report import collect_results, generate_report
from trajectorylab.utils import fingerprint, read_json, runtime_info, write_json


def worker(job: dict, config: dict, root: str, directory: str) -> str:
    root, directory = Path(root), Path(directory)
    os.environ.setdefault("MINARI_DATASETS_PATH", str(root / "artifacts/minari"))
    dataset, split, views, _ = prepare_data(config, root)
    normalizer = Normalizer.fit(
        np.concatenate([episode.observations[:-1] for episode in split["train"]])
    )
    train_run(
        job,
        config,
        dataset,
        views[job["view"]],
        split["validation"],
        normalizer,
        directory / "runs" / job["id"],
    )
    return job["id"]


def complete_suite(directory: Path, root: Path, workers: int) -> Path:
    if not 1 <= workers <= 8:
        raise ValueError("Use 1 to 8 independent workers")
    directory, root = directory.resolve(), root.resolve()
    os.environ.setdefault("MINARI_DATASETS_PATH", str(root / "artifacts/minari"))
    manifest = read_json(directory / "suite.json")
    config = manifest["config"]
    if config["training"]["device"] != "cpu":
        raise ValueError("This bounded parallel driver supports CPU experiments only")
    if manifest["implementation_digest"] != implementation_digest():
        raise ValueError("Implementation changed; start a new suite instead of mixing results")
    runtime = runtime_info()
    keys = ("python", "machine", "packages")
    if fingerprint({k: runtime[k] for k in keys}) != fingerprint(
        {k: manifest["runtime"][k] for k in keys}
    ):
        raise ValueError("Runtime differs from the initialized suite")
    dataset, _, _, audit = prepare_data(config, root)
    if audit["source_digest"] != manifest["data_audit"]["source_digest"]:
        raise ValueError("Dataset changed")
    # A driver lock prevents two instances of this helper from claiming one suite.
    # The reference serial runner predates this lock; never run both concurrently.
    lock = directory / "parallel-driver.lock"
    with lock.open("x") as stream:
        stream.write(str(os.getpid()))
    try:
        manifest.setdefault("execution_drivers", []).append(
            {
                "driver": "scripts/complete_suite.py",
                "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "workers": workers,
                "threads_per_worker": config["training"]["threads"],
                "note": "Independent CPU processes; wall times include shared-machine contention",
            }
        )
        for job in manifest["jobs"]:
            result = directory / "runs" / job["id"] / "result.json"
            if result.is_file():
                recorded = read_json(result)
                if recorded["run_id"] != job["id"]:
                    raise ValueError("Mismatched existing result")
                job["status"] = "complete"
            else:
                job["status"] = "pending"
            job.pop("error", None)
        if "references" not in manifest:
            references = {}
            names = (
                ("random", "controller")
                if dataset.metadata["env_id"] == "Pendulum-v1"
                else ("random",)
            )
            for name in names:
                result = evaluate(
                    dataset.metadata, config["evaluation"]["test_seeds"], reference=name
                )
                result.pop("states")
                references[name] = result
            manifest["references"] = references
        manifest["status"] = "running"
        write_json(directory / "suite.json", manifest)
        pending = [job for job in manifest["jobs"] if job["status"] != "complete"]
        print(f"Resuming {len(pending)} runs with {workers} workers", flush=True)
        failed = []
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            futures = {}
            for job in pending:
                job["status"] = "running"
                futures[pool.submit(worker, job.copy(), config, str(root), str(directory))] = job
            write_json(directory / "suite.json", manifest)
            for future in as_completed(futures):
                job = futures[future]
                try:
                    future.result()
                    job["status"] = "complete"
                    print(
                        f"Complete: {job['view']}/{job['algorithm']}/seed={job['seed']}", flush=True
                    )
                except Exception as error:
                    job["status"] = "failed"
                    job["error"] = f"{type(error).__name__}: {error}"
                    failed.append(job["id"])
                write_json(directory / "suite.json", manifest)
        manifest["status"] = "failed" if failed else "complete"
        write_json(directory / "suite.json", manifest)
        if failed:
            raise RuntimeError(f"Failed runs: {failed}; checkpoints remain resumable")
        collect_results(directory, manifest)
        write_json(
            root / "artifacts/latest.json", {"suite_directory": str(directory.relative_to(root))}
        )
        report = generate_report(directory, root, root / "reports" / directory.name)
        print(f"Report: {report}", flush=True)
        return report
    finally:
        lock.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    complete_suite(args.suite, Path.cwd(), args.workers)


if __name__ == "__main__":
    main()
