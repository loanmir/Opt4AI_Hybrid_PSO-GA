"""
Entry point for the Hybrid PSO-GA project 

Usage
-------

    python main.py 
    python main.py --quick
    python main.py --no-save



Output
-------


    The results of the optimization process will be saved in the 'results' directory.
    results/convergence_<benchmark_name>.png --> Convergence per benchmark
    results/summary_table.png                --> Heatmap for each bechmark and algorithm combination
    ...
"""

import argparse # for parsing command-line arguments
from pathlib import Path # for handling file paths
from benchmarks import BENCHMARKS # Importing benchmark problems
from experiments.runner import (
    run_all_algorithms,
    sweep_ga_every,
    print_results_table,
    print_best_solutions,
    save_results_txt
)
from experiments.plotting import (
    plot_multi_run_convergence,
    plot_summary_table,
    plot_ga_every_sweep
)





def parse_args():
    p = argparse.ArgumentParser(description="Hybrid PSO-GA experiment runner")
    p.add_argument("--quick",     action="store_true", help="Fast mode: 3 runs, 100 iterations")
    #p.add_argument("--easy-only", action="store_true", help="Run only the easy benchmarks")
    #p.add_argument("--hard-only", action="store_true", help="Run only the hard benchmarks")
    p.add_argument("--sweep",     action="store_true", help="Run ga_every sensitivity sweep on Hard Ackley")
    p.add_argument("--no-save",   action="store_true", help="Show plots interactively instead of saving to disk")
    return p.parse_args()





"""
# Running all three algorithms on a list of benchmarks and save the results.
def run_suite(benchmarks, label, n_runs, max_iters, out_dir):
    all_summaries = {}

    for bench in benchmarks:
        print(f"  Benchmark: {bench.name}")
        summaries = run_all_algorithms(
            bench,
            n_runs=n_runs,
            max_iters=max_iters,
            verbose=True,
        )
        all_summaries[bench.name] = summaries

        # Convergence plot
        histories_by_algorithm = {
            algorithm: s.histories for algorithm, s in summaries.items()
        }
        fname = bench.name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        save_path = out_dir / f"convergence_{fname}.png" if out_dir else None
        plot_multi_run_convergence(histories_by_algorithm, bench.name, output_path=save_path)
        print()

    # Summary heatmap
    table = {
        bname: {aname: s.mean_best for aname, s in algorithms.items()}
        for bname, algorithms in all_summaries.items()
    }
    print_results_table(all_summaries)
    print_best_solutions(all_summaries, benchmarks)
    plot_summary_table(
        table,
        maximize_flags={b.name: getattr(b, "maximize", False) for b in benchmarks},
        title=f"Final best fitness — {label} benchmarks",
        output_path=out_dir / "summary_table.png" if out_dir else None,
    )

    # Save text results
    if out_dir:
        save_results_txt(all_summaries, benchmarks, path=str(out_dir / "results.txt"))

    return all_summaries

"""



def main():
    args = parse_args()

    n_runs    = 3   if args.quick else 10
    max_iters = 100 if args.quick else 200
    base_dir  = None if args.no_save else Path("results")

    # Decide which suites to run
    if base_dir:
        base_dir.mkdir(parents=True, exist_ok=True)

    mode = "quick" if args.quick else "full"

    mode = "quick" if args.quick else "full"
    print(f"\n{'='*62}")
    print(f"  Hybrid PSO-GA  —  {mode} experiment")
    print(f"  {n_runs} runs × {max_iters} iters × 3 algorithms × 3 benchmarks")
    print(f"{'='*62}\n")


    all_summaries = {}

    for bench in BENCHMARKS:
        print(f"  Benchmark: {bench.name}")
        summaries = run_all_algorithms(
            bench,
            n_runs=n_runs,
            max_iters=max_iters,
            verbose=True,
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
    print_best_solutions(all_summaries, BENCHMARKS)
    plot_summary_table(
        table,
        maximize_flags={b.name: getattr(b, "maximize", False) for b in BENCHMARKS},
        title="Final best fitness — all benchmarks",
        output_path=base_dir / "summary_table.png" if base_dir else None,
    )

    if base_dir:
        save_results_txt(all_summaries, BENCHMARKS, path=str(base_dir / "results.txt"))

    if base_dir:
        print(f"\nDone.  All plots and results saved to ./{base_dir}/")
    else:
        print("\nDone.")


if __name__ == "__main__":
    main()