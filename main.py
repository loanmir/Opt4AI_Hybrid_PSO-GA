"""
Entry point for the Hybrid PSO-GA project.

Usage
-----

    python main.py                              Full Easy run with defaults (10 runs x 200 iters)
    python main.py --quick                      Quick Easy run with defaults (3 runs x 100 iters)
    python main.py --no-save                    Show plots interactively, don't save anything
    python main.py --use_tuned                  Use tuned hyperparameters from TUNED_CONFIGS
    python main.py --n_runs 30                  Override number of runs per (algo, benchmark)
    python main.py --max_iters 500              Override iterations per run


Output
------

The output folder is chosen automatically based on --suite and --use_tuned:

    python main.py --suite easy               ->  resultsEasy/
    python main.py --suite easy  --use_tuned  ->  resultsEasyTuned/
    python main.py --suite hard               ->  resultsHard/
    python main.py --suite hard  --use_tuned  ->  resultsHardTuned/

Each folder contains:

    results.txt                              Summary table + best solutions
    convergence_<benchmark>.png              One convergence plot per benchmark
    summary_table.png                        Heatmap of (benchmark x algorithm) results
"""

import argparse # for parsing command-line arguments
from pathlib import Path # for handling file paths
from benchmarks import EASY_BENCHMARKS, HARD_BENCHMARKS # Importing benchmark problems
from experiments.runner import (
    run_all_algorithms,
    print_results_table,
    print_best_solutions,
    save_results_txt
)
from experiments.plotting import (
    plot_multi_run_convergence,
    plot_summary_table,
)





def parse_args():
    p = argparse.ArgumentParser(description="Hybrid PSO-GA experiment runner")
    p.add_argument("--quick",     action="store_true", help="Fast mode: 3 runs, 100 iterations")
    p.add_argument("--no-save",   action="store_true", help="Show plots interactively instead of saving to disk")
    p.add_argument("--use_tuned", action="store_true", help="Use tuned hyperparameters from TUNED_CONFIGS (in experiments/runner.py).")
    p.add_argument("--n_runs",    type=int, default=None, help="Number of runs per (algo, benchmark) pair. Overrides --quick default.")
    p.add_argument("--max_iters", type=int, default=None, help="Iterations per run. Overrides --quick default.")
    p.add_argument("--suite",     choices=["easy", "hard"], default="easy", help="Which benchmark suite to run: 'easy' (Ackley/Knapsack/NAS) or 'hard' (10-D/25-item/4D versions).")
    return p.parse_args()





def main():
    args = parse_args()

    # Pick the benchmark suite
    benchmarks = EASY_BENCHMARKS if args.suite == "easy" else HARD_BENCHMARKS

    #   resultsEasy / resultsEasyTuned / resultsHard / resultsHardTuned
    if not args.no_save:
        suffix = "Tuned" if args.use_tuned else ""
        base_dir = Path(f"results{args.suite.capitalize()}{suffix}")
    else:
        base_dir = None

    n_runs    = args.n_runs    if args.n_runs    is not None else (3   if args.quick else 10)
    max_iters = args.max_iters if args.max_iters is not None else (100 if args.quick else 200)
    #base_dir  = None if args.no_save else Path("resultsTunedHard" if args.use_tuned else "resultsHard")

    # Decide which suites to run
    if base_dir:
        base_dir.mkdir(parents=True, exist_ok=True)
    mode = "quick" if args.quick else "full"
    print(f"\n{'='*62}")
    print(f"Hybrid PSO-GA  —  {mode} experiment")
    print(f"{n_runs} runs × {max_iters} iters × 3 algorithms × 3 benchmarks")
    print(f"{'='*62}\n")


    all_summaries = {}

    for bench in benchmarks:
        print(f"  Benchmark: {bench.name}")
        summaries = run_all_algorithms(
            bench,
            n_runs=n_runs,
            max_iters=max_iters,
            verbose=True,
            use_tuned=args.use_tuned,
        )
        all_summaries[bench.name] = summaries

        # Convergence plot for this benchmark
        histories_by_algo = {
            algo: s.histories for algo, s in summaries.items()
        }
        fname = bench.name.lower().replace(" ", "_")
        save_path = base_dir / f"convergence_{fname}.png" if base_dir else None
        plot_multi_run_convergence(histories_by_algo, bench.name, output_path=save_path)
        print()

    # Summary heatmap table
    table = {
        bname: {aname: s.mean_best for aname, s in algos.items()}
        for bname, algos in all_summaries.items()
    }
    print_results_table(all_summaries)
    print_best_solutions(all_summaries, benchmarks)
    plot_summary_table(
        table,
        maximize_flags={b.name: getattr(b, "maximize", False) for b in benchmarks},
        title="Final best fitness — all benchmarks",
        output_path=base_dir / "summary_table.png" if base_dir else None,
    )

    if base_dir:
        save_results_txt(all_summaries, benchmarks, path=str(base_dir / "results.txt"))

    if base_dir:
        print(f"\nDone. All plots and results saved to ./{base_dir}/")
    else:
        print("\nDone.")


if __name__ == "__main__":
    main()