"""
    Plot helpers used by the experiment runners
    Enhanced with modern, publication-ready design aesthetics.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import matplotlib.ticker as ticker

from core.pso_ga import AlgorithmResult

# Modern, vibrant color palette with high contrast
COLORS = {
    "Hybrid PSO-GA": "#2563EB",  # Deep Royal Blue
    "Pure PSO":      "#DC2626",  # Crimson Red
    "Pure GA":       "#059669",  # Emerald Green
}

LINESTYLES = {
    "Hybrid PSO-GA": "-",
    "Pure PSO":      "--",
    "Pure GA":       "-.",
}

def apply_modern_style(ax: plt.Axes) -> None:
    """Applies clean, modern chart styling by stripping clutter."""
    # Remove top and right box lines (spines)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["bottom"].set_color("#CBD5E1")
    
    # Tick adjustments
    ax.tick_params(colors="#475569", labelsize=9)
    ax.grid(True, alpha=0.2, linestyle="-", color="#94A3B8")





def save_or_show(fig: plt.Figure, path: Path | None) -> None:
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight")  # Increased DPI for crisp lines
        print(f"  Saved at {path}")
    else:
        plt.tight_layout()
        plt.show()
    plt.close(fig)






def plot_convergence(
        results: list[AlgorithmResult],
        benchmark_name: str,
        runs: int = 1,
        output_path: Path | None = None
) -> None:
    """Plots clean convergence curves with modern styling."""
    fig, ax = plt.subplots(figsize=(8.5, 4.5), facecolor="white")
    ax.set_facecolor("white")
    
    apply_modern_style(ax)
    
    for result in results:
        color = COLORS.get(result.name, "#64748B")
        ls = LINESTYLES.get(result.name, "-")
        iters = np.arange(len(result.history))
        ax.plot(iters, result.history, label=result.name, color=color, linestyle=ls, linewidth=2.2)

    ax.set_xlabel("Iteration", fontsize=10, fontweight="bold", color="#1E293B", labelpad=8)
    ax.set_ylabel("Best Fitness (lower = better)", fontsize=10, fontweight="bold", color="#1E293B", labelpad=8)
    
    suffix = "s" if runs > 1 else ""
    ax.set_title(f"{benchmark_name} Convergence ({runs} Run{suffix})", fontsize=12, fontweight="bold", color="#0F172A", pad=14, loc="left")
    
    ax.legend(frameon=True, facecolor="#F8FAFC", edgecolor="none", fontsize=9, loc="upper right")
    ax.set_yscale("symlog", linthresh=1e-2)
    
    fig.tight_layout()
    save_or_show(fig, output_path)





def plot_multi_run_convergence(
        histories_by_algorithm: dict[str, list[list[float]]],
        benchmark_name: str,
        output_path: Path | None = None
) -> None:
    """Plots clean mean + shaded standard deviation curves with fixed Y-axis scales."""
    fig, ax = plt.subplots(figsize=(8.5, 4.5), facecolor="white")
    ax.set_facecolor("white")
    
    apply_modern_style(ax)

    for algorithm_name, histories in histories_by_algorithm.items():
        arr = np.array(histories)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        iters = np.arange(arr.shape[1])
        color = COLORS.get(algorithm_name, "#64748B")
        ls = LINESTYLES.get(algorithm_name, "-")

        ax.plot(iters, mean, label=algorithm_name, color=color, linestyle=ls, linewidth=2.2)
        ax.fill_between(iters, mean - std, mean + std, color=color, alpha=0.10)

    first_history = next(iter(histories_by_algorithm.values()))[0]
    is_maximize = len(first_history) > 1 and first_history[-1] > first_history[0]
    direction = "higher = better" if is_maximize else "lower = better"
    
    ax.set_xlabel("Iteration", fontsize=10, fontweight="bold", color="#1E293B", labelpad=8)
    ax.set_ylabel(f"Best Fitness (mean ± 1 std, {direction})", fontsize=10, fontweight="bold", color="#1E293B", labelpad=8)
    
    n_runs = len(next(iter(histories_by_algorithm.values())))
    ax.set_title(f"{benchmark_name} — Benchmark Performance over {n_runs} Independent Runs", fontsize=12, fontweight="bold", color="#0F172A", pad=14, loc="left")
    
    # FIXED Y-AXIS SCALING LOGIC 
    if "knapsack" in benchmark_name.lower():
        # Purely discrete problems don't need log scaling; use a clean linear scale
        ax.set_yscale("linear")
        # Automatically space ticks nicely based on data bounds
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
    else:
        # Mixed/Continuous problems use symlog, but with explicit log-spaced formatting
        ax.set_yscale("symlog", linthresh=1e-1)
        # Force matplotlib to display standard base-10 log ticks
        ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=(1.0,)))
        ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext())
    
    ax.legend(frameon=True, facecolor="#F8FAFC", edgecolor="none", fontsize=9, loc="upper right")
    
    fig.tight_layout()
    save_or_show(fig, output_path)






def plot_summary_table(
    table: dict[str, dict[str, float]],
    maximize_flags: dict[str, bool] | None = None,
    title: str = "Final Best Fitness Matrix",
    output_path: Path | None = None 
) -> None:
    """Generates an elegant, modern heatmap table matrix."""
    benchmarks = list(table.keys())
    algorithms = list(next(iter(table.values())).keys())
    maximize_flags = maximize_flags or {b: False for b in benchmarks}

    data = np.array([[table[b][a] for a in algorithms] for b in benchmarks])

    score = np.zeros_like(data)
    for i, bname in enumerate(benchmarks):
        row = data[i]
        rmin, rmax = row.min(), row.max()
        if rmax == rmin:
            score[i] = 1.0
        elif maximize_flags.get(bname, False):
            score[i] = (row - rmin) / (rmax - rmin)
        else:
            score[i] = 1.0 - (row - rmin) / (rmax - rmin)

    n_bench = len(benchmarks)
    fig_h = max(3.5, n_bench * 1.0 + 1.2)
    fig, ax = plt.subplots(figsize=(9, fig_h), facecolor="white")
    
    
    im = ax.imshow(score, cmap="Greens", aspect="auto", vmin=0, vmax=1, alpha=0.85)

    # Clean borders out of heatmap grid
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_xticks(range(len(algorithms)))
    ax.set_xticklabels(algorithms, fontsize=10, fontweight="bold", color="#334155")
    ax.xaxis.tick_top()  # Put headers at top like a true matrix table
    
    ax.set_yticks(range(n_bench))
    ylabels = [bname + (" ↑max" if maximize_flags.get(bname, False) else " ↓min") for bname in benchmarks]
    ax.set_yticklabels(ylabels, fontsize=10, fontweight="bold", color="#334155")

    # Dynamic contrast text adjustments (Dark text on light background, light text on dark)
    for i, bname in enumerate(benchmarks):
        for j in range(len(algorithms)):
            val = data[i, j]
            text = f"{val:.3f}" if abs(val) < 100 else f"{val:.1f}"
            
            # Choose text color based on cell's performance rating background
            cell_color = "white" if score[i, j] > 0.75 else "#1E293B"
            ax.text(j, i, text, ha="center", va="center", fontsize=10, color=cell_color, fontweight="bold")
            
    ax.set_title(title, fontsize=12, fontweight="bold", color="#0F172A", pad=24, loc="center")
    
    # Custom stylized colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.7, aspect=15, pad=0.04)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(colors="#475569", labelsize=8)
    cbar.set_label("Relative Performance Scale (1.0 = Best)", fontsize=9, color="#475569", labelpad=6)
    
    fig.tight_layout()
    save_or_show(fig, output_path)

