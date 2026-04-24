# NSGA-II Reproduction

Reproducing *NSGA-II: A Fast and Elitist Multiobjective Genetic Algorithm* (Deb et al., 2002) on ZDT1–4 and ZDT6.

Run: `python3 -m src.main` — writes all plots to `results/`.

## Files

- `src/individual.py` — `Individual` class (decision vector, objectives, rank, crowding distance).
- `src/sorting.py` — `dominates` and `fast_nondominated_sort` (builds Pareto fronts F1, F2, …).
- `src/crowding.py` — `crowding_distance_assignment` and `crowded_comparison` operator.
- `src/operators.py` — `tournament_selection`, `sbx_crossover`, `polynomial_mutation`.
- `src/nsga2.py` — main NSGA-II loop: `initialize_population`, `evaluate_population`, `make_new_population`, `nsga2_step`, `run_nsga2`.
- `src/problems.py` — ZDT1–ZDT4 objective functions, bounds, true Pareto fronts, and `PROBLEMS` registry.
- `src/metrics.py` — `convergence_metric` (Υ) and `diversity_metric` (Δ) from the paper.
- `src/main.py` — orchestrates runs, trials, paper comparison, and saves all plots.
- `results/` — all generated graphs.

The same code is also packaged as a Jupyter notebook (hosted on Google Colab) for ease of reading and reproduction.
