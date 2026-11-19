"""Training-only empirical delay correction with a small calibration posterior.

Age is ``observation_tick - event_tick``; D=0 is visible at age zero.
The posterior is a working approximation, never a safety guarantee. Within a
cohort, heterogeneous public probabilities are replaced by their mean; callers
can bin public probabilities to reduce that approximation. No hidden outcomes,
scenario labels, or true calibration multipliers are accepted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Cohort:
    age: int
    public_probability: object
    arrived_conversions: int
    event_count: int | None = None

    def summary(self) -> tuple[int, float, int]:
        if np.isscalar(self.public_probability):
            p = float(self.public_probability)
            n = 1 if self.event_count is None else self.event_count
            o = self.arrived_conversions
            if not math.isfinite(p) or not 0 <= p <= 1:
                raise ValueError("public probabilities must lie in [0, 1]")
            if int(n) != n or n < 0 or int(o) != o or not 0 <= o <= n:
                raise ValueError("invalid event or arrived count")
            if int(self.age) != self.age or self.age < 0:
                raise ValueError("age must be a nonnegative integer")
            return int(n), p, int(o)
        p = np.asarray(self.public_probability, dtype=float)
        if p.ndim > 1 or not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)):
            raise ValueError("public probabilities must lie in [0, 1]")
        n = int(p.size) if self.event_count is None else self.event_count
        if int(n) != n or n < 0 or (p.ndim == 1 and p.size != n):
            raise ValueError("invalid event_count")
        o = self.arrived_conversions
        if int(o) != o or not 0 <= o <= n:
            raise ValueError("invalid arrived count")
        if int(self.age) != self.age or self.age < 0:
            raise ValueError("age must be a nonnegative integer")
        return int(n), float(p.mean()) if p.size else 0.0, int(o)


@dataclass(frozen=True)
class EmpiricalDelayCDF:
    probabilities: tuple[float, ...]
    training_sample_count: int

    @classmethod
    def fit(
        cls, delay_samples, maturity_window: int, *, split="train", all_mature=True, pseudocount=0.0
    ):
        """Fit positive-event delays from fully mature TRAINING cohorts only.

        Provenance is a caller contract: this API rejects an explicitly nontrain
        or censored dataset; it cannot independently authenticate supplied data.
        Empty training samples are rejected rather than inventing a true CDF.
        Optional total pseudocount is spread uniformly across D=0..W; it is
        an explicit regularizer, not knowledge of the evaluation delay law.
        """
        d = np.asarray(delay_samples, dtype=float)
        if split != "train" or not all_mature:
            raise ValueError("only mature training samples may fit the CDF")
        if int(maturity_window) != maturity_window or maturity_window < 0:
            raise ValueError("invalid maturity window")
        if (
            d.ndim != 1
            or not d.size
            or not np.all(np.isfinite(d))
            or np.any(d != np.floor(d))
            or np.any((d < 0) | (d > maturity_window))
        ):
            raise ValueError("invalid or empty mature delay samples")
        if not np.isfinite(pseudocount) or pseudocount < 0:
            raise ValueError("pseudocount must be finite and nonnegative")
        counts = np.bincount(d.astype(int), minlength=maturity_window + 1).astype(float)
        counts += pseudocount / len(counts)
        cumulative = np.cumsum(counts) / (len(d) + pseudocount)
        cumulative[-1] = 1.0
        return cls(tuple(cumulative), len(d))

    def __post_init__(self):
        f = np.asarray(self.probabilities)
        if (
            not f.size
            or np.any(~np.isfinite(f))
            or np.any((f < 0) | (f > 1))
            or np.any(np.diff(f) < 0)
            or f[-1] != 1
        ):
            raise ValueError("CDF must be nondecreasing and end at one")

    def __call__(self, age: int) -> float:
        if int(age) != age:
            raise ValueError("age must be integer")
        return 0.0 if age < 0 else float(self.probabilities[min(age, len(self.probabilities) - 1)])

    @property
    def maturity_window(self):
        return len(self.probabilities) - 1


@dataclass(frozen=True)
class PendingEstimate:
    arrived: int
    pending_mean: float
    total_mean: float
    calibration_mean: float


class PendingEstimator:
    """Frozen-grid scale posterior using observed arrivals and train-fit F.

    Defaults (81 scales .3..2, lognormal log-sd .5 centered at scale one)
    are development choices and must be frozen before confirmation. Stable delay
    is assumed; cold start requires separate reporting, and drift invalidates F.
    ``mature_only`` learns scale using only fully mature observable cohorts,
    then imputes immature pending counts. ``delay_corrected`` uses the MAP scale
    from all observed cohorts. ``posterior`` integrates scale uncertainty.
    """

    def __init__(
        self, delay_cdf: EmpiricalDelayCDF, method="posterior", *, scales=None, prior_log_sd=0.5
    ):
        if method not in {"naive", "mature_only", "delay_corrected", "posterior"}:
            raise ValueError("unknown method")
        self.delay_cdf, self.method = delay_cdf, method
        self.scales = np.asarray(
            np.linspace(0.3, 2.0, 81) if scales is None else scales, dtype=float
        ).copy()
        if (
            self.scales.ndim != 1
            or not self.scales.size
            or np.any(~np.isfinite(self.scales))
            or np.any(self.scales <= 0)
            or not np.isfinite(prior_log_sd)
            or prior_log_sd <= 0
        ):
            raise ValueError("scales and prior spread must be positive")
        self.log_prior = -0.5 * (np.log(self.scales) / prior_log_sd) ** 2 - np.log(self.scales)

    def _distribution(self, cohorts):
        rows = [(c, *c.summary(), self.delay_cdf(c.age)) for c in cohorts]
        # One bounded cohort-by-grid matrix; never an event-by-grid expansion.
        data = np.asarray([(n, p, o, f) for _, n, p, o, f in rows], dtype=float).reshape(-1, 4)
        n, p, o, f = (data[:, i : i + 1] for i in range(4))
        q = np.minimum(p * self.scales[None, :], 1.0)
        a = q * f
        logw = self.log_prior.copy()
        if self.method != "naive":
            selected = np.ones(len(rows), dtype=bool)
            if self.method == "mature_only":
                selected = np.asarray(
                    [c.age >= self.delay_cdf.maturity_window for c, *_ in rows], dtype=bool
                )
            # where prevents 0*log(0) from manufacturing NaNs.
            with np.errstate(divide="ignore", invalid="ignore"):
                likelihood = np.where(o > 0, o * np.log(a), 0.0)
                likelihood += np.where(n - o > 0, (n - o) * np.log1p(-a), 0.0)
            logw += likelihood[selected].sum(axis=0)
        if not np.any(np.isfinite(logw)):
            raise ValueError("observations impossible under every calibration grid point")
        weights = np.exp(logw - np.max(logw))
        weights /= weights.sum()
        if self.method in {"mature_only", "delay_corrected"}:
            maximum = int(np.argmax(weights))
            weights[:] = 0.0
            weights[maximum] = 1.0
        r = np.divide(q * (1 - f), 1 - a, out=np.zeros_like(q), where=1 - a > 0)
        pending = ((n - o) * r).sum(axis=0)
        rates = zip((n - o).ravel().astype(np.int64), r, strict=True)
        return rows, weights, pending, rates

    def estimate(self, cohorts) -> PendingEstimate:
        if self.method == "naive":
            arrived = sum(c.summary()[2] for c in cohorts)
            prior = np.exp(self.log_prior - self.log_prior.max())
            return PendingEstimate(
                arrived, 0.0, float(arrived), float(prior @ self.scales / prior.sum())
            )
        rows, weights, pending, _ = self._distribution(list(cohorts))
        arrived = sum(row[3] for row in rows)
        mean = 0.0 if self.method == "naive" else float(weights @ pending)
        return PendingEstimate(arrived, mean, arrived + mean, float(weights @ self.scales))

    def sample(self, cohorts, n_samples: int, rng=None) -> np.ndarray:
        """Return integer U; one scale draw is shared by all cohorts per draw."""
        if int(n_samples) != n_samples or n_samples < 0:
            raise ValueError("invalid sample count")
        n_samples = int(n_samples)
        if self.method == "naive":
            for c in cohorts:
                c.summary()
            return np.zeros(n_samples, dtype=np.int64)
        _, weights, _, rates = self._distribution(list(cohorts))
        rng = np.random.default_rng(rng)
        indices = rng.choice(len(weights), size=n_samples, p=weights)
        result = np.zeros(n_samples, dtype=np.int64)
        if self.method != "naive":
            for n, rate in rates:
                result += rng.binomial(n, rate[indices])
        return result

    def sample_shared(self, cohorts, n_samples: int, n_actions: int, rng=None):
        """Read-only broadcast view of the SAME past U for every candidate action."""
        if int(n_actions) != n_actions or n_actions < 1:
            raise ValueError("invalid action count")
        return np.broadcast_to(
            self.sample(cohorts, n_samples, rng)[:, None], (n_samples, n_actions)
        )
