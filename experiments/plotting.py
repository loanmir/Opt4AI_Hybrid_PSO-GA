"""
    Plot helpers used by the experiment runners

"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

from core.pso_ga import AlgorithmResult


COLORS = {
    "Hybrid PSO-GA": "#378ADD",
    "Pure PSO":      "#E24B4A",
    "Pure GA":       "#1D9E75",
}

LINESTYLES = {
    "Hybrid PSO-GA": "-",
    "Pure PSO":      "--",
    "Pure GA":       "-.",
}


# Consistent saving/showing logic for all plots
def save_or_show(fig: plt.Figure, path: Path | None) -> None:
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved at {path}")
    else:
        plt.tight_layout()
        plt.show()
    plt.close(fig)









# Convergence curves for one benchmark, 3 algorithms 
def plot_convergence(
        results: list[AlgorithmResult], # list of AlgorithmResult - one per algorithm
        benchmark_name: str, # benchmark name used for the title
        runs: int = 1, # number of runs used to produce the results
        output_path: Path | None = None # If given then plot stored there, else just shown interactively
) -> None:
    """
        Plotting converges curves (best fitness vs iteration) for one benchmark.
        If runs > 1, it will plot the mean curve with shaded area representing std deviation across runs.
    """

    fig, ax = plt.subplots(figsize=(8, 4.5))
    
    for result in results:
        color = COLORS.get(result.name, "grey")
        ls = LINESTYLES.get(result.name, "-")
        iters = np.arange(len(result.history))
        ax.plot(iters, result.history, label=result.name, color=color, linestyle=ls, linewidth=1.8)

    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel("Best fitness (lower = better)", fontsize=11)
    suffix = "s" if runs > 1 else ""
    ax.set_title(f"{benchmark_name}  —  convergence  ({runs} run{suffix})", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.set_yscale("symlog", linthresh=1e-2)   # log scale shows tail behaviour
    fig.tight_layout()
    save_or_show(fig, output_path)
















def plot_multi_run_convergence(
        histories_by_algorithm: dict[str, list[list[float]]], # {algorithm_name: [run1_history, run2_history, ...]}
        benchmark_name: str,
        output_path: Path | None = None
) -> None:
    """
        Plotting mean + standard convergence across multiple independent runs
    """

    fig, ax = plt.subplots(figsize=(8, 4.5))

    for algorithm_name, histories in histories_by_algorithm.items():
        arr = np.array(histories)           # shape (n_runs, n_iters)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        iters = np.arange(arr.shape[1])
        color = COLORS.get(algorithm_name, "grey")
        ls = LINESTYLES.get(algorithm_name, "-")

        ax.plot(iters, mean, label=algorithm_name, color=color, linestyle=ls, linewidth=1.8)
        ax.fill_between(iters, mean - std, mean + std, color=color, alpha=0.15)

    # Detect maximisation from history direction (first algorithm, first vs last value)
    first_history = next(iter(histories_by_algorithm.values()))[0]
    is_maximize = len(first_history) > 1 and first_history[-1] > first_history[0]
    direction = "higher = better" if is_maximize else "lower = better"
    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel(f"Best fitness  (mean ± 1 std,  {direction})", fontsize=11)
    n_runs = len(next(iter(histories_by_algorithm.values())))
    ax.set_title(f"{benchmark_name}  —  {n_runs} independent runs", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.set_yscale("symlog", linthresh=1e-2)
    fig.tight_layout()
    save_or_show(fig, output_path)











# Heatmap-style table summary across all benchmarks and algorithms
def plot_summary_table(
    table: dict[str, dict[str, float]], # {benchmark_name: {algorithm_name: mean_best_fitness}}
    maximize_flags: dict[str, bool] | None = None, # {benchmark_name: is_maximize(True/False)}
    title: str = "Final best fitness", # plot title
    output_path: Path | None = None 
) -> None:
    """
        Heatmap-style table with rows representing benchmarks and columns representing algorithms.
        Each cell value represents the mean best fitness across runs.
        Green = best per row
    """

    benchmarks = list(table.keys())
    algorithms = list(next(iter(table.values())).keys())
    maximize_flags = maximize_flags or {b: False for b in benchmarks} # default to minimization if maximize_flags is not provided so it is None

    data = np.array([[table[b][a] for a in algorithms] for b in benchmarks]) # shape (n_benchmarks, n_algorithms)

    # Building a normalized score matrix for coloring: 1.0 = best, 0.0 = worst (per row)
    score = np.zeros_like(data)
    for i, bname in enumerate(benchmarks):
        row = data[i]
        rmin, rmax = row.min(), row.max()
        if rmax == rmin:
            score[i] = 1.0
        elif maximize_flags.get(bname, False):
            score[i] = (row - rmin) / (rmax - rmin)        # higher = greener
        else:
            score[i] = 1.0 - (row - rmin) / (rmax - rmin)  # lower = greener

    n_bench = len(benchmarks)
    fig_h   = max(3.0, n_bench * 1.1 + 1.0)
    fig, ax = plt.subplots(figsize=(8, fig_h))
    im = ax.imshow(score, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(algorithms)))
    ax.set_xticklabels(algorithms, fontsize=10)
    ax.set_yticks(range(n_bench))

    # Add (max) / (min) label to benchmark name so it is self-explanatory
    ylabels = []
    for bname in benchmarks:
        tag = " ↑max" if maximize_flags.get(bname, False) else " ↓min"
        ylabels.append(bname + tag)
    ax.set_yticklabels(ylabels, fontsize=9)

    for i, bname in enumerate(benchmarks):
        for j in range(len(algorithms)):
            val = data[i, j]
            text = f"{val:.3f}" if abs(val) < 100 else f"{val:.1f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=9, color="black", fontweight="bold")
    ax.set_title(title + "  (green = best per row)", fontsize=11, fontweight="bold", pad=12)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Relative quality (1=best)")
    fig.tight_layout()
    save_or_show(fig, output_path)










def plot_ga_every_sweep(
        sweep_results: dict[int, list[float]], # {ga_every_value: [best_fitness_run1, best_fitness_run2, ...]}
        benchmark_name: str,
        output_path: Path | None = None
) -> None:
    """
        Plotting how the "ga_every" hyperparameter affects final solution quality.
    """

    ga_every_vals = sorted(sweep_results.keys())
    means = [np.mean(sweep_results[v]) for v in ga_every_vals]
    stds  = [np.std(sweep_results[v])  for v in ga_every_vals]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(ga_every_vals, means, yerr=stds, fmt="o-", color=COLORS["Hybrid PSO-GA"], linewidth=2, markersize=7, capsize=5)
    ax.set_xlabel("ga_every  (GA update frequency: 1=every iter, 10=rarely)", fontsize=10)
    ax.set_ylabel("Mean best fitness (lower = better)", fontsize=10)
    ax.set_title(f"Effect of GA update frequency on {benchmark_name}", fontsize=11, fontweight="bold")
    ax.set_xticks(ga_every_vals)
    ax.grid(True, alpha=0.3, linestyle=":")
    fig.tight_layout()
    save_or_show(fig, output_path)   