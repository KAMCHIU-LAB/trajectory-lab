#!/usr/bin/env python3
"""One preregistered, resource-bounded comparison of three response decoders."""
from __future__ import annotations

import argparse
import copy
import fcntl
import gc
import hashlib
import json
import resource
import shutil
import signal
import sys
import time
from pathlib import Path

START = time.monotonic()

import audit_bidding_current_response_consistency as s0  # noqa: E402
import audit_bidding_winner_belief as a  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.nn import functional as F  # noqa: E402

from trajectorylab.bidding.learning import Feedback  # noqa: E402
from trajectorylab.bidding_action_decoder import (  # noqa: E402
    ActionHead,
    DecoderModel,
    encoder_features,
)
from trajectorylab.bidding_robust_data import trajectories  # noqa: E402
from trajectorylab.bidding_robust_learning import RobustLearned  # noqa: E402

ROOT = a.ROOT
PLAN = ROOT / ".omx/plans/prd-bidding-action-decoder-v1.md"
OUTPUT = ROOT / "artifacts/bidding/action-decoder-v1"
ANCHOR = ROOT / "artifacts/bidding/robust-v1/models/response-log-11/best.pt"
S0 = ROOT / "artifacts/bidding/numerical-audits/current-response-consistency-v1"
VERIFIER = ROOT / ".omx/state/bidding-v2/current-response-consistency-independent-verification.json"
VERIFIER_HASH = "d1b4fbbdf98168fb011a6e275773835e0f19e611a4439187ea5007a23d622269"
TRAIN = tuple(range(1001, 1017))
DEV = tuple(range(2001, 2007))
ARMS = ("I", "A", "P")
DEVICE = "mps"


def preflight():
    old = a.read(ROOT / ".omx/state/bidding-robust/frozen-sources.json")
    a.require(all(a.sha(ROOT / p) == h for p, h in old.items()), "Protected source changed")
    names = [Path(__file__).resolve(), PLAN, ROOT / "scripts/audit_bidding_winner_belief.py",
             ROOT / "scripts/audit_bidding_current_response_consistency.py"]
    names += [ROOT / "src/trajectorylab" / n for n in (
        "bidding_action_decoder.py", "bidding_robust_data.py", "bidding_robust_learning.py")]
    sources = old | {str(p.relative_to(ROOT)): a.sha(p) for p in names}
    a.require(a.sha(VERIFIER) == VERIFIER_HASH, "S0 independent verification changed")
    verified = a.read(VERIFIER)
    a.require(verified["integrity_verdict"] == "PASS", "S0 integrity must pass")
    authority = a.read(ROOT / "artifacts/bidding/numerical-audits/continuation-feasibility-v1/manifest.json")
    inputs = {}
    for market in TRAIN + DEV:
        path = a.MAIN / f"markets/{market}.npz"
        expected = authority["core"]["inputs"][str(path)]
        for p, key in ((path, "npz_sha256"), (path.with_suffix(".json"), "metadata_sha256")):
            h = a.sha(p)
            a.require(h == expected[key], "Existing tape changed")
            inputs[str(p.relative_to(ROOT))] = h
    for p in [ANCHOR, VERIFIER] + [S0 / n for n in (
        "manifest.json", "contexts.json", "observations.npz", "seal.json")]:
        inputs[str(p.relative_to(ROOT))] = a.sha(p)
    seal = a.read(S0 / "seal.json")
    parent = a.read(S0 / "manifest.json")
    a.require(seal["identity"] == verified["seal_identity"], "S0 verified seal binding")
    a.require(parent["freeze"] == verified["freeze"], "S0 verified freeze binding")
    a.require(a.sha(ANCHOR) == parent["frozen"]["inputs"][str(ANCHOR.relative_to(ROOT))],
              "Anchor differs from independently checked S0 checkpoint")
    a.require(a.identity(seal["files"]) == seal["identity"], "S0 seal identity")
    a.require(all(a.sha(S0 / n) == h for n, h in seal["files"].items()), "S0 seal file")
    protocol = {"version": OUTPUT.name, "arms": ARMS, "seed": 11,
                "steps": 2000, "batch_trajectory_end_pairs": 64, "context": 24,
                "loss": "original robust log SmoothL1, action0 excluded, valid-token mean",
                "scales": [48, 48, 1, 1], "train": TRAIN, "seen_development": DEV,
                "scenes": a.SCENES, "focals": a.FOCALS, "actions": a.ACTIONS.tolist(),
                "optimizer": {"name": "AdamW", "lr": .001, "weight_decay": .0001,
                              "gradient_clip_norm": 1.},
                "device": DEVICE, "threads": 1, "wall_cpu_limit_seconds": 7200,
                "environment": {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__},
                "process_peak_limit_bytes": 8 * 2**30, "output_limit_bytes": 2 * 2**30,
                "minimum_free_bytes": 10 * 2**30, "confirmation_used": False,
                "context_change": "each valid token uses its own deployment24 prefix, common to I/A/P"}
    return {"sources": sources, "inputs": inputs, "protocol": protocol}


class Budget:
    """Persist cumulative active time; an unclosed attempt requires audit."""

    def __init__(self, freeze):
        self.path, self.freeze = OUTPUT / "budget.json", freeze
        prior = a.read(self.path) if self.path.exists() else {}
        a.require(not prior.get("running", False), "Unclosed prior attempt; audit before recovery")
        a.require(not prior or prior["freeze"] == freeze, "Resource ledger identity")
        self.prior_wall, self.prior_cpu = prior.get("wall_seconds", 0.), prior.get("cpu_seconds", 0.)
        self.peak = prior.get("peak_memory_bytes", 0)
        self.attempt = prior.get("attempt", 0) + 1
        self.check()

    def check(self, running=True):
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        self.peak = max(self.peak, int(peak if sys.platform == "darwin" else peak * 1024))
        result = {"freeze": self.freeze, "attempt": self.attempt, "running": running,
                  "wall_seconds": self.prior_wall + time.monotonic() - START,
                  "cpu_seconds": self.prior_cpu + time.process_time(),
                  "peak_memory_bytes": self.peak,
                  "output_bytes": sum(p.stat().st_size for p in OUTPUT.rglob("*") if p.is_file()),
                  "free_disk_bytes": shutil.disk_usage(OUTPUT).free}
        a.write(self.path, result)
        a.require(result["wall_seconds"] <= 7200 and result["cpu_seconds"] <= 7200, "Time budget")
        a.require(self.peak <= 8 * 2**30, "Process memory budget")
        a.require(result["output_bytes"] <= 2 * 2**30, "Output budget")
        a.require(result["free_disk_bytes"] >= 10 * 2**30, "Disk reserve")
        return result


def npz(path, **arrays):
    temp = path.with_suffix(".partial.npz")
    np.savez(temp, **arrays)
    temp.replace(path)


def save_pt(path, value):
    temp = path.with_suffix(".partial.pt")
    torch.save(value, temp)
    temp.replace(path)


def state_hash(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def fresh_head(arm, device=DEVICE):
    torch.manual_seed(11)
    torch.mps.manual_seed(11)
    return ActionHead(arm, a.ACTIONS).to(device)


def get_anchor(device=DEVICE):
    policy = RobustLearned(ANCHOR, a.ACTIONS, 24, device)
    policy.model.eval().requires_grad_(False)
    return policy


@torch.no_grad()
def build_cache(anchor, freeze, budget):
    directory = OUTPUT / "cache"
    directory.mkdir(exist_ok=True)
    stamps = {}
    for market in TRAIN + DEV:
        path = directory / f"{market}.npz"
        stamp = path.with_suffix(".json")
        if stamp.exists():
            meta = a.read(stamp)
            a.require(meta["freeze"] == freeze and a.sha(path) == meta["sha256"], "Cache changed")
        else:
            a.require(not path.exists(), "Incomplete cache needs audit")
            runs = trajectories([a.MAIN / f"markets/{market}.npz"], a.ACTIONS, list(a.SCENES))
            a.require(len(runs) == 300, "Cache behavior coverage")
            states = np.asarray([r["states"] for r in runs])
            labels = np.asarray([r["labels"] for r in runs])
            common = np.empty((300, 48, 141), dtype=np.float32)
            pmean = np.empty((300, 48), dtype=np.float32)
            for tick in range(48):
                start = max(0, tick + 1 - 24)
                for lo in range(0, 300, 64):
                    x = torch.as_tensor((states[lo:lo+64, start:tick+1] - anchor.mean) / anchor.std,
                                        device=DEVICE)
                    steps = torch.arange(start, tick+1, device=DEVICE)[None].expand(len(x), -1)
                    mask = torch.ones(steps.shape, device=DEVICE, dtype=torch.bool)
                    features = encoder_features(anchor.model, x, steps, mask)[:, -1]
                    common[lo:lo+64, tick] = features.cpu().numpy()
                    # Use the identical affine recovery as the deployment adapter.
                    p = x[:, -1, 9] * float(anchor.std[9]) + float(anchor.mean[9])
                    pmean[lo:lo+64, tick] = p.cpu().numpy()
                if tick % 8 == 0:
                    budget.check()
            metadata = np.asarray([[market, r["data_diagnostics"]["focal"],
                                    a.SCENES.index(r["data_diagnostics"]["scenario"]),
                                    r["data_diagnostics"]["behavior"], t]
                                   for r in runs for t in range(48)], dtype=np.int32)
            a.require(np.allclose(pmean, states[..., 9], rtol=2e-6, atol=2e-7), "Public p recovery")
            a.require(np.isfinite(common).all() and np.isfinite(pmean).all(), "Finite cache")
            npz(path, common=common.reshape(-1, 141), pmean=pmean.reshape(-1),
                labels=labels.reshape(-1, 4, 11), metadata=metadata)
            meta = {"freeze": freeze, "sha256": a.sha(path), "rows": 14400,
                    "max_pmean_affine_recovery_error": float(np.max(abs(pmean-states[..., 9]))),
                    "context": "each tick own last24; actual timesteps; original norm layer"}
            a.write(stamp, meta)
            del runs, states, labels, common, pmean
            gc.collect()
            print(f"cache market={market} rows=14400", flush=True)
        stamps[str(market)] = meta
        budget.check()
    a.write(OUTPUT / "cache-identity.json", stamps)
    return a.identity(stamps)


def dataset(markets):
    parts = []
    for market in markets:
        with np.load(OUTPUT / f"cache/{market}.npz", allow_pickle=False) as data:
            parts.append({k: data[k] for k in data.files})
    return {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}


def batches():
    rng = np.random.default_rng(11)
    values, offsets = [], [0]
    for _ in range(2000):
        ids = []
        for _ in range(64):
            run = int(rng.integers(4800))
            end = int(rng.integers(1, 49))
            ids.extend(range(run*48+max(0, end-24), run*48+end))
        values.extend(ids)
        offsets.append(len(values))
    return np.asarray(values, dtype=np.int64), np.asarray(offsets, dtype=np.int64), rng.bit_generator.state


def fit(arm, train, indices, offsets, rng_state, identity, budget):
    head = fresh_head(arm)
    initial = state_hash(head)
    optimizer = torch.optim.AdamW(head.parameters(), lr=.001, weight_decay=.0001)
    path = OUTPUT / f"{arm}-checkpoint.pt"
    step, history = 0, []

    def snapshot():
        save_pt(path, {"identity": identity, "arm": arm, "step": step,
                       "initial_state_sha256": initial, "model": head.state_dict(),
                       "optimizer": optimizer.state_dict(), "torch_rng": torch.get_rng_state(),
                       "mps_rng": torch.mps.get_rng_state(), "sampler_rng": copy.deepcopy(rng_state),
                       "next_minibatch": step, "history": history})

    if path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=False)
        a.require(saved["identity"] == identity and saved["arm"] == arm, "Checkpoint identity")
        a.require(saved["initial_state_sha256"] == initial, "Initialization mismatch")
        a.require(0 <= saved["step"] <= 2000, "Step budget")
        head.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["torch_rng"])
        torch.mps.set_rng_state(saved["mps_rng"])
        a.require(saved["sampler_rng"] == rng_state, "Sampler RNG mismatch")
        step, history = saved["step"], saved["history"]
        a.require(saved["next_minibatch"] == step, "Checkpoint minibatch position")
    else:
        snapshot()
    scales = torch.tensor([48., 48., 1., 1.], device=DEVICE)[None, :, None]
    while step < 2000:
        ids = indices[offsets[step]:offsets[step+1]]
        x, p, y = (torch.as_tensor(train[k][ids], device=DEVICE) for k in ("common", "pmean", "labels"))
        prediction = head(x, p)[..., 1:]
        error = F.smooth_l1_loss(torch.log1p(prediction*scales), torch.log1p(y[..., 1:]*scales), reduction="none")
        loss = error.mean((-1, -2)).sum() / len(ids)
        a.require(bool(torch.isfinite(loss)), "Nonfinite loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 1.)
        a.require(bool(torch.isfinite(norm)), "Nonfinite gradient")
        optimizer.step()
        step += 1
        if step % 100 == 0:
            a.require(all(bool(torch.isfinite(p).all()) for p in head.parameters()), "Nonfinite weight")
            history.append({"step": step, "training_loss": float(loss.detach())})
            snapshot()
            budget.check()
            print(f"train arm={arm} step={step} loss={float(loss.detach()):.6f}", flush=True)
    return head, {"initial_state_sha256": initial, "final_state_sha256": state_hash(head),
                  "parameters": sum(p.numel() for p in head.parameters()), "step": step,
                  "history": history}


@torch.no_grad()
def predict(head, data, budget, decoder=None):
    chunks = []
    for lo in range(0, len(data["common"]), 1024):
        x = torch.as_tensor(data["common"][lo:lo+1024], device=DEVICE)
        if decoder is None:
            p = torch.as_tensor(data["pmean"][lo:lo+1024], device=DEVICE)
            value = head(x, p)
        else:
            raw = decoder(x[..., :128]).reshape(-1, 4, 10)
            value = F.pad(F.softplus(raw).cumsum(-1), (1, 0))
        chunks.append(value.cpu().numpy())
        if lo % 10240 == 0:
            budget.check()
    value = np.concatenate(chunks)
    a.require(np.isfinite(value).all() and np.all(value >= 0), "Invalid prediction")
    a.require(np.all(value[..., 0] == 0) and np.all(np.diff(value, axis=-1) >= 0), "Curve constraints")
    return value


def prediction_metrics(pred, data):
    error = ((pred.astype(np.float64)-data["labels"])[..., 1:]**2).mean(-1)
    zero = (data["labels"][..., 1:] == 0).mean(-1)
    rows = []
    meta = data["metadata"]
    for market in DEV:
        for si, scene in enumerate(a.SCENES):
            selected = (meta[:, 0] == market) & (meta[:, 2] == si)
            a.require(int(selected.sum()) == 2880, "Development cell coverage")
            rows.append({"market": market, "scene": scene, "contexts": int(selected.sum()),
                         "mse": error[selected].mean(0).tolist(), "zero_label_rate": zero[selected].mean(0).tolist()})
    return {"rows": rows, "all_mse": np.mean([r["mse"] for r in rows], axis=0).tolist(),
            "by_scene": {s: np.mean([r["mse"] for r in rows if r["scene"] == s], axis=0).tolist() for s in a.SCENES}}


@torch.no_grad()
def s0_features(anchor):
    contexts = a.read(S0 / "contexts.json")
    common, pmean = [], []
    with np.load(S0 / "observations.npz", allow_pickle=False) as data:
        for i, context in enumerate(contexts):
            length = context["length"]
            x = torch.as_tensor((data["states"][i, :, :length]-anchor.mean)/anchor.std, device=DEVICE)
            step = torch.as_tensor(np.broadcast_to(data["steps"][i, :length], (4, length)).copy(), device=DEVICE)
            mask = torch.ones(step.shape, dtype=torch.bool, device=DEVICE)
            common.append(encoder_features(anchor.model, x, step, mask)[:, -1].cpu().numpy())
            pmean.append((x[:, -1, 9]*float(anchor.std[9])+float(anchor.mean[9])).cpu().numpy())
    return contexts, {"common": np.asarray(common).reshape(-1, 141), "pmean": np.asarray(pmean).reshape(-1)}


def pair_metrics(predictions, contexts):
    pred = predictions.reshape(96, 4, 4, 11)
    groups = {s: [] for s in ("initial_shift", "registered_midshift", "additional_shock")}
    for i, context in enumerate(contexts):
        for si, ai, bi in s0.pairs():
            groups[s0.stratum(context["tick"], si)].append((pred[i, si, :, bi].astype(float)-pred[i, 0, :, ai])**2)
    combined = groups["initial_shift"]+groups["registered_midshift"]
    a.require(len(combined) == 240, "S0 selected pair coverage")
    return {"strata": {key: {"pairs": len(values), "gap_mse": np.mean(values, axis=0).tolist()} for key, values in groups.items()},
            "combined_pairs": 240, "combined_gap_mse": np.mean(combined, axis=0).tolist()}


def ratio(value, reference):
    return float(value/reference) if reference != 0 else None


def prediction_gate(metrics, pairs):
    flags = {}
    for other in ("A", "I"):
        for h in (2, 3):
            r = ratio(metrics["P"]["all_mse"][h], metrics[other]["all_mse"][h])
            flags[f"tail{h}_MSE_P/{other}_le_.95"] = r is not None and r <= .95
            r = ratio(metrics["P"]["by_scene"]["calibrated"][h], metrics[other]["by_scene"]["calibrated"][h])
            flags[f"nominal_tail{h}_MSE_P/{other}_le_1.02"] = r is not None and r <= 1.02
            flags[f"S0_tail{h}_gap_P_lt_{other}"] = pairs["P"]["combined_gap_mse"][h] < pairs[other]["combined_gap_mse"][h]
    return {"passed": all(flags.values()), "flags": flags}


def policy_metrics(rows):
    result = {}
    for reference in ("FB", "A", "I"):
        scenes, per_market = {}, {}
        for scene in a.SCENES:
            p = [r for r in rows if r["arm"] == "P" and r["scene"] == scene]
            b = [r for r in rows if r["arm"] == reference and r["scene"] == scene]
            scenes[scene] = {"score_ratio": ratio(np.mean([r["score"] for r in p]), np.mean([r["score"] for r in b])),
                             "conversion_ratio": ratio(np.mean([r["conversions"] for r in p]), np.mean([r["conversions"] for r in b])),
                             "P_violations": sum(r["violation"] for r in p), "reference_violations": sum(r["violation"] for r in b)}
        for market in DEV:
            ratios = [ratio(np.mean([r["score"] for r in rows if r["market"] == market and r["scene"] == s and r["arm"] == "P"]),
                            np.mean([r["score"] for r in rows if r["market"] == market and r["scene"] == s and r["arm"] == reference])) for s in a.SCENES]
            per_market[str(market)] = float(np.mean(ratios)) if all(r is not None for r in ratios) else None
        ratios = [v["score_ratio"] for v in scenes.values()]
        result[reference] = {"scenes": scenes, "macro": float(np.mean(ratios)) if all(r is not None for r in ratios) else None, "per_market_macro": per_market}
    def at_least(x, threshold):
        return x is not None and x >= threshold
    fb = result["FB"]
    flags = {"over50": at_least(fb["scenes"]["over_50"]["score_ratio"], 1.03),
             "under25": at_least(fb["scenes"]["under_25"]["score_ratio"], 1.03),
             "macro": at_least(fb["macro"], 1.03),
             "markets": sum(x is not None and x > 1 for x in fb["per_market_macro"].values()) >= 4}
    for scene, cell in fb["scenes"].items():
        flags[f"protect_{scene}"] = at_least(cell["score_ratio"], .98) and at_least(cell["conversion_ratio"], .97) and cell["P_violations"] <= cell["reference_violations"]
    for other in ("A", "I"):
        flags[f"increment_vs_{other}"] = at_least(result[other]["macro"], 1.005)
        flags[f"protect_vs_{other}"] = all(at_least(v["score_ratio"], .98) for v in result[other]["scenes"].values())
    return {"comparisons": result, "flags": flags, "passed": all(flags.values())}


def policy_replay(heads, budget):
    policies = {"FB": Feedback(a.ACTIONS, .3, 8.)}
    for arm, head in heads.items():
        policy = get_anchor("cpu")
        policy.model = DecoderModel(policy.model, copy.deepcopy(head).cpu(), policy.mean, policy.std).eval()
        policies[arm] = policy
    rows, switched = [], 0
    with (OUTPUT / "policy-decisions.jsonl").open("w") as log:
        for market in DEV:
            tape = a.Tape(a.MAIN / f"markets/{market}.npz")
            for scene in a.SCENES:
                for fi, focal in enumerate(a.FOCALS):
                    budget_value, cpa = float(tape.arrays["budgets"][fi]), float(tape.arrays["cpas"][fi])
                    for arm, policy in policies.items():
                        if hasattr(policy, "reset"):
                            policy.reset()
                        cost, conv, history, used, rewards = 0., 0, [], [], []
                        previous = (0., 0, 0., 0)
                        for tick in range(48):
                            state = a.observation(tape, fi, tick, cost, conv, previous, a.scenario_bias(scene, tick))
                            history.append(state)
                            action, diagnostic = policy(history, used, rewards, budget_value, cpa)
                            fb_action, _ = policies["FB"]([state], [], [], budget_value, cpa)
                            switched += int(arm == "P" and action != fb_action)
                            tc, cv, wins, blocked = tape.settle(fi, tick, action*a.scenario_bias(scene, tick), budget_value-cost)
                            diagnostic["budget_blocked"] = blocked
                            if hasattr(policy, "observe"):
                                policy.observe(tc, cv, diagnostic)
                            rewards.append(a.score(cost+tc, conv+cv, cpa)-a.score(cost, conv, cpa))
                            used.append(action)
                            log.write(json.dumps({"market": market, "scene": scene, "focal": focal, "arm": arm,
                                                  "tick": tick, "state": state.tolist(), "action": action,
                                                  "own_state_FB_action": fb_action, "tick_cost": tc,
                                                  "tick_C": cv, "wins": wins, "budget_blocked": bool(blocked)})+"\n")
                            cost, conv = cost+tc, conv+cv
                            previous = tc, cv, action, wins
                        a.require(cost <= budget_value+1e-6, "Policy hard budget")
                        rows.append({"market": market, "scene": scene, "focal": focal, "arm": arm,
                                     "cost": cost, "conversions": conv, "budget": budget_value, "cpa": cpa,
                                     "score": a.score(cost, conv, cpa), "violation": int(cost > cpa*conv+1e-8)})
            del tape
            gc.collect()
            budget.check()
            print(f"policy market={market}", flush=True)
    a.require(len(rows) == 720, "Policy coverage")
    a.write(OUTPUT / "policy-terminals.json", rows)
    result = policy_metrics(rows)
    result["P_non_FB_actions"] = switched
    result["flags"]["non_FB_action"] = switched > 0
    result["passed"] = all(result["flags"].values())
    return result


def run(frozen, freeze, resume):
    if resume:
        a.require(a.read(OUTPUT / "manifest.json")["freeze"] == freeze, "Resume freeze")
        a.require(not (OUTPUT / "summary.json").exists(), "Completed study cannot be rerun")
        seal = a.read(OUTPUT / "seal.json")
        a.require(a.identity(seal["files"]) == seal["identity"], "Recovery seal identity")
        a.require(all(a.sha(OUTPUT / n) == h for n, h in seal["files"].items()), "Recovery sealed file changed")
    else:
        OUTPUT.mkdir(parents=True, exist_ok=False)
        a.write(OUTPUT / "manifest.json", {"freeze": freeze, "frozen": frozen, "frozen_before_fit": True})
    torch.set_num_threads(1)
    a.require(torch.backends.mps.is_available(), "MPS required; no undeclared device substitution")
    budget = Budget(freeze)
    try:
        anchor = get_anchor()
        encoder_hash = state_hash(anchor.model)
        cache_id = build_cache(anchor, freeze, budget)
        indices, offsets, rng_state = batches()
        sampler_path = OUTPUT / "minibatches.npz"
        if not sampler_path.exists():
            npz(sampler_path, indices=indices, offsets=offsets)
        with np.load(sampler_path) as saved:
            a.require(np.array_equal(saved["indices"], indices) and np.array_equal(saved["offsets"], offsets), "Minibatch stream")
        sampler_id = a.sha(sampler_path)
        identity = a.identity({"freeze": freeze, "cache": cache_id, "sampler": sampler_id})
        train = dataset(TRAIN)
        a.require(len(train["common"]) == 230400, "Training rows")
        initial_a, initial_p = fresh_head("A"), fresh_head("P")
        a.require(state_hash(initial_a) == state_hash(initial_p), "A/P initialization")
        x = torch.as_tensor(train["common"][:32], device=DEVICE)
        p = torch.as_tensor(train["pmean"][:32], device=DEVICE)
        ia, ip = initial_a.interval_inputs(x, p), initial_p.interval_inputs(x, p)
        a.require(torch.equal(ia[..., :143], ip[..., :143]), "A/P common inputs")
        a.require(bool((ia[..., 143] == 0).all()), "A zero product slot")
        a.require(torch.equal(ip[..., 143], p[:, None]*initial_p.midpoints[None, :]), "P public product slot")
        a.write(OUTPUT / "training-identity.json", {"identity": identity, "cache": cache_id,
                 "sampler_sha256": sampler_id, "A_P_initial_sha256": state_hash(initial_a),
                 "input_evidence": {"A_P_first143_bitwise_equal": True,
                                    "A_slot_zero": True, "P_first_row_slot": ip[0, :, 143].cpu().tolist(),
                                    "first_row_public_pmean": float(p[0]),
                                    "interval_midpoints_div3": ((a.ACTIONS[1:]+a.ACTIONS[:-1])/6).tolist()}})
        del initial_a, initial_p, x, p, ia, ip
        heads, training = {}, {}
        for arm in ARMS:
            heads[arm], training[arm] = fit(arm, train, indices, offsets, rng_state, identity, budget)
        a.require(training["A"]["initial_state_sha256"] == training["P"]["initial_state_sha256"], "Matched A/P init")
        a.require([training[n]["parameters"] for n in ARMS] == [13720, 13700, 13700], "Parameter budget")
        a.require(state_hash(anchor.model) == encoder_hash, "Encoder changed")
        a.write(OUTPUT / "training.json", training)
        del train
        gc.collect()
        dev = dataset(DEV)
        a.require(len(dev["common"]) == 86400, "Development rows")
        contexts, paired = s0_features(anchor)
        metrics, gaps = {}, {}
        for arm in ARMS + ("old-anchor",):
            decoder = anchor.model.decoder if arm == "old-anchor" else None
            head = heads.get(arm)
            pred = predict(head, dev, budget, decoder)
            metrics[arm] = prediction_metrics(pred, dev)
            npz(OUTPUT / f"{arm}-dev-predictions.npz", predictions=pred)
            pair_pred = predict(head, paired, budget, decoder)
            gaps[arm] = pair_metrics(pair_pred, contexts)
            npz(OUTPUT / f"{arm}-S0-predictions.npz", predictions=pair_pred.reshape(96, 4, 4, 11))
        gate = prediction_gate(metrics, gaps)
        a.write(OUTPUT / "prediction-metrics.json", {"models": metrics, "S0": gaps, "gate": gate})
        print(json.dumps({"prediction_gate": gate}), flush=True)
        policy = policy_replay(heads, budget) if gate["passed"] else None
        a.require(preflight() == frozen, "Source/input changed during execution")
        resources = budget.check()
        summary = {"status": "complete", "freeze": freeze, "prediction_gate": gate,
                   "policy": policy, "policy_skipped": policy is None,
                   "candidate_pass": gate["passed"] and policy is not None and policy["passed"],
                   "independent_confirmation": False, "generative_contribution": False,
                   "resources": resources, "integrity_review": "pending independent verification"}
        a.write(OUTPUT / "summary.json", summary)
        print(json.dumps({"candidate_pass": summary["candidate_pass"], "policy_skipped": policy is None}), flush=True)
    except BaseException as error:
        a.write(OUTPUT / f"failure-attempt-{budget.attempt}.json", {"error": type(error).__name__, "message": str(error), "freeze": freeze})
        raise
    finally:
        try:
            budget.check(running=False)
        except BaseException as error:
            if (OUTPUT / "summary.json").exists():
                invalid = a.read(OUTPUT / "summary.json")
                invalid.update(status="invalid_or_incomplete", candidate_pass=False,
                               final_resource_error=str(error))
                a.write(OUTPUT / "summary.json", invalid)
            raise
        finally:
            files = {str(p.relative_to(OUTPUT)): a.sha(p) for p in OUTPUT.rglob("*") if p.is_file() and p.name != "seal.json"}
            a.write(OUTPUT / "seal.json", {"files": files, "identity": a.identity(files)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--freeze")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    frozen = preflight()
    freeze = a.identity(frozen)
    print(json.dumps({"preflight": "PASS", "freeze": freeze, "fitted": False}), flush=True)
    if args.run:
        a.require(args.freeze == freeze, "Requires reviewed current freeze")
        def interrupt(_signum, _frame):
            raise KeyboardInterrupt("Requested process stop")
        signal.signal(signal.SIGTERM, interrupt)
        with (OUTPUT.parent / ".action-decoder-v1.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            run(frozen, freeze, args.resume)
    else:
        a.require(not args.resume, "Resume requires run")


if __name__ == "__main__":
    main()
