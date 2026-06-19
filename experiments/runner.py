"""
    Utility functions for running multiple indepedent experiments of each algorithm on each benchmark problem, and for saving the results.
"""

from __future__ import annotations
import time 
import numpy as np
import io
from pathlib import Path
from dataclasses import dataclass
from core.pso_ga import (
    run_hybrid, run_pure_pso, run_pure_ga, AlgorithmResult
)

from benchmarks import BENCHMARKS


@dataclass
class MultiRunSummary:
    algorithm_name: str
    benchmark_name: str
    n_runs: int
    mean_best: float
    std_best: float
    best_of_runs: float
    mean_time_sec: float
    histories: list[list[float]] # List of convergence histories for each run
    maximize: bool = False
    best_x_cont: np.ndarray | None = None   # solution vector from best run
    best_x_disc: np.ndarray | None = None




def run_all_algorithms(
    benchmark,
    # *,
    n_runs: int = 10,
    n_particles: int = 30,
    max_iters: int = 200,
    ga_every: int = 3,
    w: float = 0.7,
    c1: float = 1.5,
    c2: float = 1.5,
    p_cross: float = 0.7,
    p_mut: float = 0.1,
    verbose: bool = True,
) -> dict[str, MultiRunSummary]:
    """
        Running Hybrid PSO-GA, Pure PSO and Pure GA on one benchmark for n_runs independent runs each 

        It returns a dictionary mapping algorithm names to their MultiRunSummary dataclass instances, which contain the results and convergence histories for each algorithm.
    """

    shared = dict(
        n_continuous=benchmark.n_continuous,
        discrete_options=benchmark.discrete_options,
        cont_lb=benchmark.cont_lb,
        cont_ub=benchmark.cont_ub,
        n_particles=n_particles,
        max_iters=max_iters,
    )

    fitness_fn = benchmark.fitness
    maximize = getattr(benchmark, "maximize", False)

    configs = [
        ("Hybrid PSO-GA",
         lambda: run_hybrid(fitness_fn, **shared,
                            ga_every=ga_every, w=w, c1=c1, c2=c2,
                            p_cross=p_cross, p_mut=p_mut,
                            maximize=maximize)),
        ("Pure PSO",
         lambda: run_pure_pso(fitness_fn, **shared,
                              w=w, c1=c1, c2=c2,
                              maximize=maximize)),
        ("Pure GA",
         lambda: run_pure_ga(fitness_fn, **shared,
                             p_cross=p_cross, p_mut=p_mut,
                             maximize=maximize)),
    ]

    summaries: dict[str, MultiRunSummary] = {}

    for algo_name, runner in configs:
        if verbose:
            print(f"  Running {algo_name} × {n_runs} on {benchmark.name} …",
                  end="", flush=True)
        bests, histories, times = [], [], []
        best_result: AlgorithmResult | None = None

        for _ in range(n_runs):
            t0 = time.perf_counter()
            result: AlgorithmResult = runner()
            times.append(time.perf_counter() - t0)
            bests.append(result.best_fitness)
            histories.append(result.history)
            # Track which run produced the best solution
            if best_result is None:
                best_result = result
            elif maximize and result.best_fitness > best_result.best_fitness:
                best_result = result
            elif not maximize and result.best_fitness < best_result.best_fitness:
                best_result = result

        summaries[algo_name] = MultiRunSummary(
            algorithm_name=algo_name,
            benchmark_name=benchmark.name,
            n_runs=n_runs,
            mean_best=float(np.mean(bests)),
            std_best=float(np.std(bests)),
            best_of_runs=float(np.max(bests) if maximize else np.min(bests)),
            mean_time_sec=float(np.mean(times)),
            histories=histories,
            maximize=maximize,
            best_x_cont=best_result.best_continuous,
            best_x_disc=best_result.best_discrete,
        )
        if verbose:
            s = summaries[algo_name]
            print(f"  done  "
                  f"mean={s.mean_best:.4f} ± {s.std_best:.4f}  "
                  f"({s.mean_time_sec:.1f}s/run)")

    return summaries





# TAKE A LOOK ALSO AT THIS FUNCTION!! AGAIN!!!

def sweep_ga_every(
    benchmark,
    ga_every_values: list[int],
    # *,
    n_runs: int = 10,
    n_particles: int = 30,
    max_iters: int = 200,
    verbose: bool = True,
) -> dict[int, list[float]]:
    """
    Run the Hybrid PSO-GA for each value of ga_every and collect best-fitness
    distributions.  Used to produce the sensitivity plot.

    Returns {ga_every: [best_fit_run1, best_fit_run2, ...]}
    """
    results: dict[int, list[float]] = {}
    fitness_fn = benchmark.fitness

    for gae in ga_every_values:
        if verbose:
            print(f"  ga_every={gae:2d} …", end="", flush=True)
        bests = []
        for _ in range(n_runs):
            r = run_hybrid(
                fitness_fn,
                n_continuous=benchmark.n_continuous,
                discrete_options=benchmark.discrete_options,
                cont_lb=benchmark.cont_lb,
                cont_ub=benchmark.cont_ub,
                n_particles=n_particles,
                max_iters=max_iters,
                ga_every=gae,
            )
            bests.append(r.best_fitness)
        results[gae] = bests
        if verbose:
            print(f"  mean={np.mean(bests):.4f}")

    return results










def print_results_table(all_summaries: dict[str, dict[str, MultiRunSummary]]) -> None:
    """
    Print a comparison table to the terminal.

    
    Takes all_summaries : {benchmark_name: {algo_name: MultiRunSummary}}
    """
    benchmarks = list(all_summaries.keys())
    algorithms      = list(next(iter(all_summaries.values())).keys())

    col_w = 26
    header = f"{'Benchmark':<22}" + "".join(f"{a:>{col_w}}" for a in algorithms)
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    for bname in benchmarks:
        row = f"{bname:<22}"
        for aname in algorithms:
            s = all_summaries[bname][aname]
            row += f"  {s.mean_best:>8.4f} ± {s.std_best:<8.4f}"
        print(row)

    print("=" * len(header))
    print("Values: mean best fitness ± std  across runs")
    print("  Minimisation benchmarks: lower = better")
    print("  Maximisation benchmarks: higher = better  (Mixed Knapsack)\n")








def print_best_solutions(all_summaries: dict[str, dict[str, MultiRunSummary]],
                         benchmarks_list: list) -> None:
    """
    Print the actual best solution vectors found by each algorithm on each
    benchmark 
    """
    # Build a lookup for benchmark metadata 
    bench_meta = {b.name: b for b in benchmarks_list}

    for bname, algorithm_summaries in all_summaries.items():
        bench = bench_meta.get(bname)
        maximize = getattr(bench, "maximize", False)
        direction = "higher = better" if maximize else "lower = better"

        print(f"\n{'─'*60}")
        print(f"  {bname}   ({direction})")
        print(f"{'─'*60}")

        for aname, s in algorithm_summaries.items():
            print(f"\n  [{aname}]")
            print(f"    Fitness  -->  mean: {s.mean_best:.6f}  "
                  f"std: {s.std_best:.6f}  "
                  f"best single run: {s.best_of_runs:.6f}")

            if s.best_x_cont is not None:
                cont_str = "  ".join(f"{v:.6f}" for v in s.best_x_cont)
                print(f"    x_cont  -->  [{cont_str}]")

            if s.best_x_disc is not None:
                disc_str = "  ".join(str(v) for v in s.best_x_disc)
                print(f"    x_disc -->[{disc_str}]")

                # Add human-readable interpretation per benchmark
                if "Knapsack" in bname and bench is not None:
                    selected = [i for i, v in enumerate(s.best_x_disc) if v == 1]
                    total_val = sum(bench.VALUES[i] for i in selected)  # Defined in KNapsack class
                    total_wt  = sum(bench.WEIGHTS[i] for i in selected)
                    cap_scale = float(s.best_x_cont[0]) if (s.best_x_cont is not None and len(s.best_x_cont) > 0) else 1.0
                    capacity  = bench.CAPACITY * cap_scale #if hasattr(bench, "CAPACITY") else bench.BASE_CAPACITY * cap_scale
                    #cap = getattr(bench, "CAPACITY", getattr(bench, "BASE_CAPACITY", 1.5))
                    cap = bench.CAPACITY
                    print(f" Items selected: {selected}")
                    print(f" Total value:    {total_val:.2f}"
                          f"(capacity used: {total_wt:.2f} / {cap:.2f})")

                elif "NAS" in bname and bench is not None:
                    lr      = float(s.best_x_cont[0])
                    dropout = float(s.best_x_cont[1])
                    n_layers = int(s.best_x_disc[0]) + 1
                    act_map  = {0: "relu", 1: "tanh", 2: "sigmoid"}
                    act      = act_map.get(int(s.best_x_disc[1]), "?")
                    print(f"lr={lr:.6f}  dropout={dropout:.4f}"
                          f"layers={n_layers}  activation={act}")

                elif "Ackley" in bname:
                    print(f"(optimum is x_cont = all zeros)")





# TAKE AGAIN A LOOK AT THIS! --> Recall if this function is needed, Maybe just plots are needed?!?!?

def save_results_txt(
    all_summaries: dict[str, dict[str, MultiRunSummary]],
    benchmarks_list: list,
    path: str = "resultsEasy/results.txt",
) -> None:
    """
    Save the full results — fitness table + best solution vectors — to a
    plain-text file
    """
    

    buf = io.StringIO()

    # --- Header ---
    buf.write("=" * 70 + "\n")
    buf.write("  HYBRID PSO-GA  —  EXPERIMENT RESULTS\n")
    buf.write("=" * 70 + "\n\n")

    benchmarks = list(all_summaries.keys())
    algorithms      = list(next(iter(all_summaries.values())).keys())

    # --- Summary table ---
    buf.write("SUMMARY TABLE  (mean best fitness ± std across runs)\n")
    buf.write("  Minimisation benchmarks (Ackley, NAS): lower = better\n")
    buf.write("  Maximisation benchmark  (Knapsack):    higher = better\n\n")

    col_w = 28
    header = f"{'Benchmark':<22}" + "".join(f"{a:>{col_w}}" for a in algorithms)
    buf.write("=" * len(header) + "\n")
    buf.write(header + "\n")
    buf.write("=" * len(header) + "\n")

    for bname in benchmarks:
        row = f"{bname:<22}"
        for aname in algorithms:
            s = all_summaries[bname][aname]
            row += f"  {s.mean_best:>9.6f} ± {s.std_best:<9.6f}"
        buf.write(row + "\n")

    buf.write("=" * len(header) + "\n\n")

    # --- Best solution details ---
    buf.write("BEST SOLUTION DETAILS  (from the single best run)\n")
    bench_meta = {b.name: b for b in benchmarks_list}

    for bname, algo_summaries in all_summaries.items():
        bench = bench_meta.get(bname)
        maximize = getattr(bench, "maximize", False)
        direction = "higher = better" if maximize else "lower = better"
        buf.write(f"\n{'─'*60}\n")
        buf.write(f"  {bname}   ({direction})\n")
        buf.write(f"{'─'*60}\n")

        for aname, s in algo_summaries.items():
            buf.write(f"\n  [{aname}]\n")
            buf.write(f"    Fitness     mean={s.mean_best:.6f}"
                      f"std={s.std_best:.6f}  "
                      f"best={s.best_of_runs:.6f}\n")

            if s.best_x_cont is not None:
                cont_str = "  ".join(f"{v:.6f}" for v in s.best_x_cont)
                buf.write(f"    x_cont    [{cont_str}]\n")

            if s.best_x_disc is not None:
                disc_str = "  ".join(str(v) for v in s.best_x_disc)
                buf.write(f"    x_disc    [{disc_str}]\n")

                if "Knapsack" in bname and bench is not None:
                    selected  = [i for i, v in enumerate(s.best_x_disc) if v == 1]
                    total_val = sum(bench.VALUES[i] for i in selected)
                    total_wt  = sum(bench.WEIGHTS[i] for i in selected)
                    cap_scale = float(s.best_x_cont[0]) if (s.best_x_cont is not None and len(s.best_x_cont) > 0) else 1.0
                    capacity  = bench.CAPACITY * cap_scale #if hasattr(bench, "CAPACITY") else bench.BASE_CAPACITY * cap_scale
                    #cap = getattr(bench, "CAPACITY", getattr(bench, "BASE_CAPACITY", 1.5))
                    cap = bench.CAPACITY
                    buf.write(f" Items selected: {selected}\n")
                    buf.write(f" Total value:    {total_val:.2f}  "
                              f"(weight: {total_wt:.2f} / capacity: {cap:.2f})\n")

                elif "NAS" in bname and bench is not None:
                    lr       = float(s.best_x_cont[0])
                    dropout  = float(s.best_x_cont[1])
                    n_layers = int(s.best_x_disc[0]) + 1
                    act_map  = {0: "relu", 1: "tanh", 2: "sigmoid"}
                    act      = act_map.get(int(s.best_x_disc[1]), "?")
                    buf.write(f" Architecture:   lr={lr:.6f}  "
                              f"dropout={dropout:.4f}  "
                              f"layers={n_layers}  activation={act}\n")

    buf.write("\n" + "=" * 70 + "\n")

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(buf.getvalue(), encoding="utf-8")
    print(f"  Saved → {out}")