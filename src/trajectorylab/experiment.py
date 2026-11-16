"""An auditable experiment runner: explicit budgets, manifests, and checkpoints."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np
import torch

from trajectorylab.algorithms import Learner, Replay, Sequences
from trajectorylab.data import (
    Dataset,
    Normalizer,
    generate_pendulum,
    import_minari,
    select_view,
    split_episodes,
    transitions,
)
from trajectorylab.evaluation import Policy, SupportDiagnostic, evaluate
from trajectorylab.utils import fingerprint, read_json, runtime_info, seed_everything, write_json


def implementation_digest() -> str:
    h = hashlib.sha256()
    for name in (
        "algorithms.py",
        "models.py",
        "data.py",
        "evaluation.py",
        "experiment.py",
        "config.py",
    ):
        h.update(name.encode())
        h.update(Path(__file__).with_name(name).read_bytes())
    return h.hexdigest()


def prepare_data(config: dict, root: Path) -> tuple:
    spec = {
        k: v
        for k, v in config["data"].items()
        if k not in ("views", "transition_budget", "budget_tolerance", "split_seed")
    }
    source_key = fingerprint(spec)[:12]
    directory = root / "artifacts" / "datasets" / source_key
    if (directory / "dataset.json").exists():
        dataset = Dataset.load(directory)
    else:
        print(f"Preparing {spec['kind']} source ({spec['source_episodes']} episodes)", flush=True)
        if spec["kind"] == "pendulum":
            dataset = generate_pendulum(spec["source_episodes"], spec["source_seed"])
        else:
            dataset = import_minari(spec["minari_id"], spec["source_episodes"], spec["source_seed"])
        dataset.save(directory)
    split = split_episodes(dataset.episodes, config["data"]["split_seed"])
    views = {
        name: select_view(
            split["train"],
            name,
            config["data"]["transition_budget"],
            config["data"]["split_seed"],
            config["data"].get("budget_tolerance", 0.05),
        )
        for name in config["data"]["views"]
    }
    audits = {}
    for name, episodes in views.items():
        table = transitions(episodes)
        episode_returns = [e.total_return for e in episodes]
        audits[name] = {
            "episode_ids": [e.id for e in episodes],
            "episodes": len(episodes),
            "transitions": len(table["actions"]),
            "budget_fraction": len(table["actions"]) / config["data"]["transition_budget"],
            "return_quantiles": np.quantile(episode_returns, [0, 0.25, 0.5, 0.75, 1]).tolist(),
            "terminal_count": int(table["terminated"].sum()),
            "timeout_count": int(table["truncated"].sum()),
            "action_saturation_fraction": float((np.abs(table["actions"]) >= 0.99).mean()),
            "digest": fingerprint([(e.id, e.digest) for e in episodes]),
        }
    audit = {
        "source_metadata": dataset.metadata,
        "source_digest": dataset.digest,
        "dataset_directory": str(directory.relative_to(root)),
        "split_episode_ids": {k: [e.id for e in episodes] for k, episodes in split.items()},
        "views": audits,
        "split_seed": config["data"]["split_seed"],
        "support_scaling": "common normalizer fitted on the source training pool only",
    }
    return dataset, split, views, audit


def save_checkpoint(path: Path, state: dict) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def load_checkpoint(path: Path, expected_run_id: str | None = None) -> dict:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if expected_run_id is not None and state["run_id"] != expected_run_id:
        raise ValueError("Checkpoint configuration/data/code fingerprint does not match")
    return state


def train_run(
    job: dict,
    config: dict,
    dataset: Dataset,
    episodes: list,
    validation: list,
    diagnostic_normalizer: Normalizer,
    output: Path,
) -> dict:
    train = config["training"]
    seed_everything(job["seed"], train["threads"])
    rng = np.random.default_rng(job["seed"])
    normalizer = Normalizer.fit(np.concatenate([e.observations[:-1] for e in episodes]))
    observation_dim, action_dim = episodes[0].observations.shape[1], episodes[0].actions.shape[1]
    target_return = float(
        np.quantile([e.total_return for e in episodes], train["target_return_quantile"])
    )
    if max(len(e) for e in episodes) > train["max_timestep"]:
        raise ValueError("Dataset episode exceeds max_timestep; increase the explicit vocabulary")
    learner = Learner(job["algorithm"], observation_dim, action_dim, train)
    initial = torch.cat([p.detach().flatten().cpu() for p in learner.actor.parameters()]).clone()
    if job["algorithm"] == "dt":
        sampler = Sequences(
            episodes, normalizer, train["context"], train["return_scale"], train["device"]
        )
    else:
        sampler = Replay(episodes, normalizer, train["device"])
    policy = Policy(learner, normalizer, train, target_return)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "checkpoint.pt"
    history, start_step, prior_elapsed = [], 0, 0.0
    if checkpoint.exists():
        saved = load_checkpoint(checkpoint, job["id"])
        learner.load_state(saved["learner"])
        rng.bit_generator.state = saved["numpy_rng"]
        torch.set_rng_state(saved["torch_rng"])
        start_step, history, prior_elapsed = (
            saved["step"],
            saved["history"],
            saved["elapsed_seconds"],
        )
        initial = saved["initial_actor"]
        print(f"  resuming at update {start_step}", flush=True)
    start = time.perf_counter()

    def state(step: int) -> dict:
        return {
            "run_id": job["id"],
            "algorithm": job["algorithm"],
            "step": step,
            "learner": learner.state(),
            "numpy_rng": rng.bit_generator.state,
            "torch_rng": torch.get_rng_state(),
            "initial_actor": initial,
            "normalizer": normalizer.state(),
            "target_return": target_return,
            "training": train,
            "observation_dim": observation_dim,
            "action_dim": action_dim,
            "source_metadata": dataset.metadata,
            "history": history,
            "elapsed_seconds": prior_elapsed + time.perf_counter() - start,
        }

    for step in range(start_step + 1, train["steps"] + 1):
        metrics = learner.update(sampler.sample(rng, train["batch_size"]))
        if step % train["log_every"] == 0 or step == train["steps"]:
            row = {
                "step": step,
                **metrics,
                "elapsed_seconds": prior_elapsed + time.perf_counter() - start,
            }
            if step % train["eval_every"] == 0 or step == train["steps"]:
                validation_result = evaluate(
                    dataset.metadata, config["evaluation"]["validation_seeds"], policy
                )
                row["validation_return"] = validation_result["return_mean"]
                print(
                    f"  update {step}/{train['steps']} val_return={row['validation_return']:.2f}",
                    flush=True,
                )
            history.append(row)
        if step % train["checkpoint_every"] == 0:
            save_checkpoint(checkpoint, state(step))
    save_checkpoint(checkpoint, state(train["steps"]))
    write_json(output / "history.json", history)
    # Final test is run only after the fixed training budget, never used for selection.
    final = evaluate(dataset.metadata, config["evaluation"]["test_seeds"], policy)
    eval_states = final.pop("states")
    train_states = np.concatenate([e.observations[:-1] for e in episodes])
    validation_states = np.concatenate([e.observations[:-1] for e in validation])
    support = SupportDiagnostic(
        train_states, validation_states, diagnostic_normalizer, config["data"]["split_seed"]
    )
    final["support"] = support.summarize(eval_states)
    parameters = torch.cat([p.detach().flatten().cpu() for p in learner.actor.parameters()])
    if not torch.isfinite(parameters).all():
        raise FloatingPointError("Non-finite actor parameters")
    result = {
        "run_id": job["id"],
        "algorithm": job["algorithm"],
        "view": job["view"],
        "seed": job["seed"],
        "updates": train["steps"],
        "test": final,
        "actor_parameter_delta_l2": float(torch.linalg.vector_norm(parameters - initial)),
        "wall_seconds": prior_elapsed + time.perf_counter() - start,
        "target_return_from_training_data": target_return,
        "final_training_metrics": history[-1] if history else {},
        "checkpoint_policy": "last update; no test-based checkpoint selection",
    }
    np.savez_compressed(output / "test_states.npz", states=eval_states)
    write_json(output / "result.json", result)
    return result


def run_suite(config: dict, root: Path) -> Path:
    seed_everything(0, config["training"]["threads"])
    dataset, split, views, audit = prepare_data(config, root)
    code_hash = implementation_digest()
    runtime = runtime_info()
    runtime_key = fingerprint({k: runtime[k] for k in ("python", "machine", "packages")})
    suite_id = fingerprint(
        {
            "config": config,
            "data": audit["source_digest"],
            "code": code_hash,
            "runtime": runtime_key,
        }
    )[:12]
    directory = root / "artifacts" / "suites" / f"{config['suite']['name']}-{suite_id}"
    directory.mkdir(parents=True, exist_ok=True)
    jobs = []
    for view in config["data"]["views"]:
        for algorithm in config["suite"]["algorithms"]:
            for seed in config["suite"]["seeds"]:
                identity = {
                    "algorithm": algorithm,
                    "view_digest": audit["views"][view]["digest"],
                    "seed": seed,
                    "training": config["training"],
                    "evaluation": config["evaluation"],
                    "implementation": code_hash,
                    "source_digest": audit["source_digest"],
                    "runtime": runtime_key,
                }
                jobs.append(
                    {
                        "id": fingerprint(identity)[:16],
                        "algorithm": algorithm,
                        "view": view,
                        "seed": seed,
                        "status": "pending",
                    }
                )
    manifest = {
        "schema_version": 1,
        "suite_id": suite_id,
        "config": config,
        "implementation_digest": code_hash,
        "runtime": runtime,
        "data_audit": audit,
        "jobs": jobs,
        "status": "running",
        "expected_runs": len(jobs),
    }
    for job in jobs:
        path = directory / "runs" / job["id"] / "result.json"
        if path.exists():
            existing = read_json(path)
            if existing["run_id"] != job["id"]:
                raise ValueError("Mismatched result identity")
            job["status"] = "complete"
    write_json(directory / "suite.json", manifest)
    references = {}
    for name in (
        ("random", "controller") if dataset.metadata["env_id"] == "Pendulum-v1" else ("random",)
    ):
        reference = evaluate(dataset.metadata, config["evaluation"]["test_seeds"], reference=name)
        reference.pop("states")
        references[name] = reference
    manifest["references"] = references
    diagnostic_normalizer = Normalizer.fit(
        np.concatenate([e.observations[:-1] for e in split["train"]])
    )
    for number, job in enumerate(jobs, 1):
        print(
            f"[{number}/{len(jobs)}] {job['view']}/{job['algorithm']}/seed={job['seed']} ({job['status']})",
            flush=True,
        )
        if job["status"] == "complete":
            continue
        job["status"] = "running"
        write_json(directory / "suite.json", manifest)
        try:
            train_run(
                job,
                config,
                dataset,
                views[job["view"]],
                split["validation"],
                diagnostic_normalizer,
                directory / "runs" / job["id"],
            )
        except Exception as error:
            job["status"] = "failed"
            job["error"] = f"{type(error).__name__}: {error}"
            manifest["status"] = "failed"
            write_json(directory / "suite.json", manifest)
            raise
        job["status"] = "complete"
        write_json(directory / "suite.json", manifest)
    manifest["status"] = "complete"
    write_json(directory / "suite.json", manifest)
    write_json(
        root / "artifacts" / "latest.json", {"suite_directory": str(directory.relative_to(root))}
    )
    return directory


def evaluate_checkpoint(checkpoint: Path, seeds: list[int]) -> dict:
    saved = load_checkpoint(checkpoint)
    learner = Learner(
        saved["algorithm"], saved["observation_dim"], saved["action_dim"], saved["training"]
    )
    learner.load_state(saved["learner"])
    policy = Policy(
        learner,
        Normalizer.from_state(saved["normalizer"]),
        saved["training"],
        saved["target_return"],
    )
    result = evaluate(saved["source_metadata"], seeds, policy)
    result.pop("states")
    return result
