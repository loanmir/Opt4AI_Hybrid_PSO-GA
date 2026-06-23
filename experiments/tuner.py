"""
Random search over hyperparameters for the three runners in core.pso_ga.

Usage:
    python -m experiments.tuner --benchmark HardNAS --algo hybrid
    python -m experiments.tuner --benchmark HardAckley --algo pso
    python -m experiments.tuner --benchmark HardKnapsack --algo ga
"""

from __future__ import annotations
import argparse
import numpy as np
from dataclasses import dataclass, asdict
from typing import Callable

from benchmarks.functions import Ackley, HardAckley, Knapsack, HardKnapsack, NAS, HardNAS
from core.pso_ga import run_hybrid, run_pure_pso, run_pure_ga, AlgorithmResult


@dataclass
class TrialResult:
    params: dict
    best_fitness: float
    std_across_seeds: float
    n_evaluations: int


# Default search space. Tuples of (low, high). Integer ranges
# are sampled as ints, float ranges as floats.
DEFAULT_PARAM_SPACE = {
    "w":        (0.4, 0.9),     # PSO inertia
    "c1":       (1.0, 2.5),     # PSO cognitive
    "c2":       (1.0, 2.5),     # PSO social
    "p_cross":  (0.5, 0.95),    # GA crossover
    "p_mut":    (0.01, 0.20),   # GA mutation
    "ga_every": (1, 10),        # integer: GA every N iters
}

# Population size and iteration count are FIXED (not tuned) to keep
# the search space small. If you want to tune these too, add them here.
FIXED_KWARGS = {
    "n_particles": 30,
    "max_iters":   300,
}


def _sample_params(space: dict) -> dict:
    out = {}
    for k, (lo, hi) in space.items():
        if isinstance(lo, int) and isinstance(hi, int):
            out[k] = int(np.random.randint(lo, hi + 1))
        else:
            out[k] = float(np.random.uniform(lo, hi))
    return out


def random_search(
    runner: Callable,
    benchmark,
    n_trials: int = 30,
    n_seeds: int = 3,
    param_space: dict | None = None,
) -> list[TrialResult]:
    """Run n_trials random configs, each averaged over n_seeds."""

    space = param_space or DEFAULT_PARAM_SPACE
    trials: list[TrialResult] = []

    for trial_idx in range(n_trials):
        params = _sample_params(space)
        seed_fits: list[float] = []
        last_result: AlgorithmResult | None = None

        for seed in range(n_seeds):
            np.random.seed(seed)
            result: AlgorithmResult = runner(
                fitner_fn=benchmark.fitness,
                n_continuous=benchmark.n_continuous,
                discrete_options=benchmark.discrete_options,
                cont_lb=benchmark.cont_lb,
                cont_ub=benchmark.cont_ub,
                maximize=benchmark.maximize,
                **FIXED_KWARGS,
                **params,
            )
            seed_fits.append(result.best_fitness)
            last_result = result

        trials.append(TrialResult(
            params=params,
            best_fitness=float(np.mean(seed_fits)),
            std_across_seeds=float(np.std(seed_fits)),
            n_evaluations=last_result.n_evaluations,
        ))

    # For maximization, higher is better; for minimization, lower is better.
    trials.sort(key=lambda t: -t.best_fitness if benchmark.maximize else t.best_fitness)
    return trials


BENCHMARKS = {
    "ackley":       Ackley(),
    "hardackley":   HardAckley(),
    "knapsack":     Knapsack(),
    "hardknapsack": HardKnapsack(),
    "nas":          NAS(),
    "hardnas":      HardNAS(),
}

ALGOS = {
    "hybrid": run_hybrid,
    "pso":    run_pure_pso,
    "ga":     run_pure_ga,
}


def main():
    parser = argparse.ArgumentParser(description="Random search tuner for the three optimizers.")
    parser.add_argument("--benchmark", required=True, choices=list(BENCHMARKS.keys()))
    parser.add_argument("--algo",      required=True, choices=list(ALGOS.keys()))
    parser.add_argument("--n_trials",  type=int, default=30)
    parser.add_argument("--n_seeds",   type=int, default=3)
    args = parser.parse_args()

    benchmark = BENCHMARKS[args.benchmark]
    runner    = ALGOS[args.algo]
    direction = "maximize" if benchmark.maximize else "minimize"

    print(f"Tuning {args.algo} on {args.benchmark} ({direction})")
    print(f"  trials={args.n_trials} seeds/trial={args.n_seeds}")
    print(f"  fixed: {FIXED_KWARGS}")
    print()

    trials = random_search(
        runner=runner,
        benchmark=benchmark,
        n_trials=args.n_trials,
        n_seeds=args.n_seeds,
    )

    print(f"Top 5 configs ({direction}):")
    for t in trials[:5]:
        print(f"  fit={t.best_fitness:.6f}  std={t.std_across_seeds:.4f}  {t.params}")
    print()
    print(f"Best config:\n  {trials[0].params}")


if __name__ == "__main__":
    main()