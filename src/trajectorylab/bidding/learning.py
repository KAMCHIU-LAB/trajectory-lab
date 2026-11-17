"""Small causal policies and response forecasts; all fitting uses training markets."""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from trajectorylab.models import CausalBlock, DecisionTransformer
from trajectorylab.utils import write_json

from .market import Tape, feedback_action, observation, scenario_bias, score


def rollout(
    tape: Tape, fi: int, actions: np.ndarray, scenario: str, chooser, *, retain: bool = False
) -> dict:
    a = tape.arrays
    budget, cpa = float(a["budgets"][fi]), float(a["cpas"][fi])
    states, used, rewards, diagnostics = [], [], [], []
    cost, conv, wins, previous_score = 0.0, 0, 0, 0.0
    previous = (0.0, 0, 0.0, 0)
    for tick in range(48):
        bias = scenario_bias(scenario, tick)
        state = observation(tape, fi, tick, cost, conv, previous, bias)
        states.append(state)
        action, diagnostic = chooser(states, used, rewards, budget, cpa)
        if not np.isfinite(action) or not np.any(np.isclose(actions, action, rtol=0, atol=1e-8)):
            raise RuntimeError("Policy returned an invalid action")
        tick_cost, tick_conv, tick_wins, budget_blocked = tape.settle(
            fi, tick, action * bias, budget - cost
        )
        diagnostic["budget_blocked"] = budget_blocked
        cost += tick_cost
        conv += tick_conv
        wins += tick_wins
        current_score = score(cost, conv, cpa)
        rewards.append(current_score - previous_score)
        previous_score = current_score
        previous = (tick_cost, tick_conv, action, tick_wins)
        used.append(action)
        diagnostics.append(diagnostic)
        if hasattr(chooser, "observe"):
            chooser.observe(tick_cost, tick_conv, diagnostic)
    if cost > budget + 1e-6:
        raise RuntimeError("Hard-budget settlement exceeded remaining funds")
    result = {
        "cost": cost,
        "conversions": conv,
        "exposures": wins,
        "budget": budget,
        "target_cpa": cpa,
        "cpa": cost / conv if conv else None,
        "cpa_debt": cost - cpa * conv,
        "violation": int(cost > cpa * conv + 1e-8),
        "zero_conversions": int(conv == 0),
        "budget_utilization": cost / budget,
        "score": score(cost, conv, cpa),
        "budget_blocked_ticks": sum(d["budget_blocked"] for d in diagnostics),
    }
    predicted = [d for d in diagnostics if "predicted_tick_cost" in d]
    if predicted:
        result.update(
            predicted_cost=sum(d["predicted_tick_cost"] for d in predicted),
            predicted_conversions=sum(d["predicted_tick_conversions"] for d in predicted),
            forecast_feasible_fraction=float(np.mean([d["forecast_feasible"] for d in predicted])),
        )
    if retain:
        result.update(
            states=np.asarray(states, dtype=np.float32),
            actions=np.asarray(used, dtype=np.float32),
            rewards=np.asarray(rewards, dtype=np.float32),
        )
    return result


class Feedback:
    def __init__(self, actions, gain=0.7, prior=3.0):
        self.actions, self.gain, self.prior = actions, gain, prior

    def __call__(self, states, used, rewards, budget, cpa):
        return feedback_action(states[-1], self.actions, self.gain, self.prior, budget, cpa), {}


def trajectories(paths: list[Path], actions: np.ndarray) -> list[dict]:
    result = []
    for path in paths:
        tape = Tape(path)
        idx = np.array([tape.index(k) for k in actions])
        for fi in range(len(tape.arrays["focals"])):
            raw_cost = tape.arrays["costs"][fi][:, idx]
            raw_conv = tape.arrays["conversions"][fi][:, idx]
            remaining_cost = raw_cost[::-1].cumsum(axis=0)[::-1]
            remaining_conv = raw_conv[::-1].cumsum(axis=0)[::-1]
            budget, cpa = tape.arrays["budgets"][fi], tape.arrays["cpas"][fi]
            labels = np.stack(
                [
                    raw_cost / budget,
                    raw_conv * cpa / budget,
                    remaining_cost / budget,
                    remaining_conv * cpa / budget,
                ],
                axis=1,
            )
            policies = [Feedback(actions, gain, 3.0) for gain in (0.3, 0.7, 1.2)]
            for fixed in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
                policies.append(
                    lambda s, u, r, b, c, k=fixed: (
                        float(actions[np.abs(actions - k).argmin()]),
                        {},
                    )
                )
            for policy in policies:
                run = rollout(tape, fi, actions, "calibrated", policy, retain=True)
                reward = run.pop("rewards")
                run["returns"] = reward[::-1].cumsum()[::-1].copy()
                run["labels"] = labels.astype(np.float32)
                result.append(run)
    return result


class ResponseModel(nn.Module):
    """Monotone mean curves for current and remaining-horizon costs/conversions."""

    def __init__(self, dimension: int, action_count: int, width: int, layers: int, heads: int):
        super().__init__()
        self.action_count = action_count
        self.embedding = nn.Linear(dimension, width)
        self.time = nn.Embedding(48, width)
        self.blocks = nn.ModuleList([CausalBlock(width, heads) for _ in range(layers)])
        self.norm = nn.LayerNorm(width)
        self.decoder = nn.Linear(width, 4 * (action_count - 1))
        with torch.no_grad():
            self.decoder.bias.reshape(4, action_count - 1)[:2].fill_(-5.0)
            self.decoder.bias.reshape(4, action_count - 1)[2:].fill_(-2.0)

    def forward(self, states, steps, mask):
        hidden = self.embedding(states) + self.time(steps)
        for block in self.blocks:
            hidden = block(hidden, mask)
        raw = self.decoder(self.norm(hidden)).reshape(*states.shape[:2], 4, self.action_count - 1)
        curve = F.softplus(raw).cumsum(dim=-1)
        return F.pad(curve, (1, 0), value=0.0)


def _batch(data, rng, batch_size, context, mean, std, return_scale, device):
    dimension = len(mean)
    n_actions = data[0]["labels"].shape[-1]
    states = np.zeros((batch_size, context, dimension), dtype=np.float32)
    actions = np.zeros((batch_size, context, 1), dtype=np.float32)
    returns = np.zeros((batch_size, context), dtype=np.float32)
    labels = np.zeros((batch_size, context, 4, n_actions), dtype=np.float32)
    steps = np.zeros((batch_size, context), dtype=np.int64)
    mask = np.zeros((batch_size, context), dtype=bool)
    for i in range(batch_size):
        trajectory = data[int(rng.integers(len(data)))]
        end = int(rng.integers(1, 49))
        start = max(0, end - context)
        length = end - start
        states[i, :length] = (trajectory["states"][start:end] - mean) / std
        actions[i, :length, 0] = trajectory["actions"][start:end] * 2 / 3 - 1
        returns[i, :length] = trajectory["returns"][start:end] / return_scale
        labels[i, :length] = trajectory["labels"][start:end]
        steps[i, :length] = np.arange(start, end)
        mask[i, :length] = True
    values = (states, actions, returns, labels, steps, mask)
    return tuple(torch.as_tensor(x, device=device) for x in values)


def _loss(model, kind, batch):
    states, actions, returns, labels, steps, mask = batch
    if kind == "dt":
        prediction = model(states, actions, returns, steps, mask)
        return ((prediction - actions).square().squeeze(-1) * mask).sum() / mask.sum()
    prediction = model(states, steps, mask)
    scales = torch.tensor([48.0, 48.0, 1.0, 1.0], device=states.device)[None, None, :, None]
    errors = F.smooth_l1_loss(
        torch.log1p(prediction * scales), torch.log1p(labels * scales), reduction="none"
    )
    return (errors.mean((-1, -2)) * mask).sum() / mask.sum()


def fit(
    kind: str,
    seed: int,
    config: dict,
    train: list[dict],
    validation: list[dict],
    directory: Path,
    identity: str,
) -> dict:
    opts = config["training"]
    device = opts["device"]
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Requested MPS is unavailable; declare a different device explicitly")
    torch.set_num_threads(opts["threads"])
    torch.manual_seed(seed)
    if device == "mps":
        torch.mps.manual_seed(seed)
    rng = np.random.default_rng(seed)
    states = np.concatenate([x["states"] for x in train])
    mean, std = states.mean(axis=0), np.maximum(states.std(axis=0), 0.05)
    target = float(np.quantile([x["score"] for x in train], 0.9))
    return_scale = max(1.0, target)
    args = dict(width=opts["width"], layers=opts["layers"], heads=opts["heads"])
    model = (
        DecisionTransformer(len(mean), 1, max_timestep=48, **args)
        if kind == "dt"
        else ResponseModel(len(mean), len(config["study"]["actions"]), **args)
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=opts["learning_rate"], weight_decay=1e-4)
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / "checkpoint.pt"
    best = directory / "best.pt"
    history = []
    step, best_loss, total_seconds = 0, float("inf"), 0.0
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved["identity"] != identity:
            raise RuntimeError("Training source/configuration identity mismatch")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["torch_rng"])
        if device == "mps":
            torch.mps.set_rng_state(saved["mps_rng"])
        rng.bit_generator.state = saved["numpy_rng"]
        step, best_loss, total_seconds = saved["step"], saved["best_loss"], saved["seconds"]
        history = saved["history"]
    started = time.perf_counter()
    metadata = {
        "identity": identity,
        "kind": kind,
        "seed": seed,
        "mean": mean,
        "std": std,
        "return_scale": return_scale,
        "target_return": target,
        "architecture": args,
        "parameters": sum(p.numel() for p in model.parameters()),
    }

    def save(path, payload):
        temp = path.with_suffix(".partial.pt")
        torch.save(payload, temp)
        os.replace(temp, path)

    while step < opts["steps"]:
        model.train()
        batch = _batch(
            train, rng, opts["batch_size"], opts["context"], mean, std, return_scale, device
        )
        loss = _loss(model, kind, batch)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(norm):
            raise RuntimeError("Nonfinite gradient")
        optimizer.step()
        step += 1
        if step % opts["log_every"] == 0 or step == opts["steps"]:
            model.eval()
            validation_rng = np.random.default_rng(800001)
            with torch.no_grad():
                losses = [
                    float(
                        _loss(
                            model,
                            kind,
                            _batch(
                                validation,
                                validation_rng,
                                opts["batch_size"],
                                opts["context"],
                                mean,
                                std,
                                return_scale,
                                device,
                            ),
                        )
                    )
                    for _ in range(8)
                ]
            val = float(np.mean(losses))
            history.append(
                {
                    "step": step,
                    "training_loss": float(loss.detach()),
                    "validation_loss": val,
                    "seconds": total_seconds + time.perf_counter() - started,
                }
            )
            print(
                f"train kind={kind} seed={seed} step={step} validation_loss={val:.6f}", flush=True
            )
            write_json(directory / "history.json", history)
            if val < best_loss:
                best_loss = val
                save(
                    best,
                    metadata | {"model": model.state_dict(), "step": step, "validation_loss": val},
                )
        if step % opts["checkpoint_every"] == 0 or step == opts["steps"]:
            save(
                checkpoint,
                metadata
                | {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "best_loss": best_loss,
                    "history": history,
                    "seconds": total_seconds + time.perf_counter() - started,
                    "torch_rng": torch.get_rng_state(),
                    "mps_rng": torch.mps.get_rng_state() if device == "mps" else None,
                    "numpy_rng": rng.bit_generator.state,
                },
            )
    return {
        "kind": kind,
        "seed": seed,
        "best_validation_loss": best_loss,
        "steps": step,
        "parameters": metadata["parameters"],
        "seconds": total_seconds + time.perf_counter() - started,
    }


class Learned:
    def __init__(
        self,
        path: Path,
        actions: np.ndarray,
        context: int,
        device: str,
        corrected: bool = False,
        prior: float = 3.0,
        hard_cpa: bool = False,
    ):
        saved = torch.load(path, map_location="cpu", weights_only=False)
        self.kind = saved["kind"]
        self.actions, self.context, self.device = actions, context, device
        self.mean, self.std = saved["mean"], saved["std"]
        self.target, self.scale = saved["target_return"], saved["return_scale"]
        args = saved["architecture"]
        self.model = (
            DecisionTransformer(len(self.mean), 1, max_timestep=48, **args)
            if self.kind == "dt"
            else ResponseModel(len(self.mean), len(actions), **args)
        ).to(device)
        self.model.load_state_dict(saved["model"])
        self.model.eval()
        self.corrected, self.prior = corrected, prior
        self.hard_cpa = hard_cpa
        self.actual_cost, self.actual_conv, self.pred_cost, self.pred_conv = 0.0, 0.0, 0.0, 0.0

    def reset(self):
        self.actual_cost, self.actual_conv, self.pred_cost, self.pred_conv = 0.0, 0.0, 0.0, 0.0

    @torch.no_grad()
    def __call__(self, states, used, rewards, budget, cpa):
        end = len(states)
        start = max(0, end - self.context)
        x = torch.as_tensor(
            (np.asarray(states[start:]) - self.mean) / self.std, device=self.device
        )[None]
        steps = torch.arange(start, end, device=self.device)[None]
        mask = torch.ones((1, end - start), dtype=torch.bool, device=self.device)
        if self.kind == "dt":
            past = np.array(used + [0.0], dtype=np.float32)[start:] * 2 / 3 - 1
            rtg = (
                np.array(
                    [self.target - sum(rewards[:i]) for i in range(start, end)], dtype=np.float32
                )
                / self.scale
            )
            output = self.model(
                x,
                torch.as_tensor(past, device=self.device)[None, :, None],
                torch.as_tensor(rtg, device=self.device)[None],
                steps,
                mask,
            )[0, -1, 0]
            proposed = (float(output) + 1) * 1.5
            if self.corrected:
                state = states[-1]
                cost = float(state[3]) * budget
                conv = float(state[4]) * budget / cpa
                ratio = np.clip((conv + self.prior) * cpa / (cost + self.prior * cpa), 0.25, 2.0)
                proposed *= ratio
            return float(self.actions[np.abs(self.actions - proposed).argmin()]), {}
        curves = self.model(x, steps, mask)[0, -1].cpu().numpy()
        scale = np.array([budget, budget / cpa, budget, budget / cpa])[:, None]
        curves = curves * scale
        future_cost, future_conv = curves[2].copy(), curves[3].copy()
        if self.corrected:
            cost_ratio = np.clip(
                (self.actual_cost + self.prior * cpa) / (self.pred_cost + self.prior * cpa),
                0.25,
                4.0,
            )
            conv_ratio = np.clip(
                (self.actual_conv + self.prior) / (self.pred_conv + self.prior), 0.25, 4.0
            )
            future_cost *= cost_ratio
            future_conv *= conv_ratio
        spent, conv = float(states[-1][3]) * budget, float(states[-1][4]) * budget / cpa
        affordable = future_cost <= max(0, budget - spent)
        feasible = affordable & (spent + future_cost <= cpa * (conv + future_conv))
        if not self.hard_cpa:
            candidates = np.flatnonzero(affordable)
            predicted_conv = conv + future_conv[candidates]
            predicted_cost = spent + future_cost[candidates]
            utility = predicted_conv * np.minimum(
                1.0, (cpa * predicted_conv / np.maximum(predicted_cost, 1e-12)) ** 2
            )
            ai = int(candidates[np.argmax(utility)]) if len(candidates) else 0
        elif feasible.any():
            candidates = np.flatnonzero(feasible)
            ai = int(candidates[np.argmax(future_conv[candidates])])
        else:
            debt = spent + future_cost - cpa * (conv + future_conv)
            candidates = np.flatnonzero(affordable)
            ai = int(candidates[np.argmin(debt[candidates])]) if len(candidates) else 0
        return float(self.actions[ai]), {
            "predicted_tick_cost": float(curves[0, ai]),
            "predicted_tick_conversions": float(curves[1, ai]),
            "forecast_feasible": bool(feasible[ai]),
        }

    def observe(self, cost, conv, diagnostic):
        # Unthrottled forecast targets cannot be calibrated against censored ticks.
        if diagnostic.get("budget_blocked", False):
            return
        self.actual_cost += cost
        self.actual_conv += conv
        self.pred_cost += diagnostic.get("predicted_tick_cost", 0.0)
        self.pred_conv += diagnostic.get("predicted_tick_conversions", 0.0)
