# NSGA-II Reproduction

A from-scratch Python implementation of **NSGA-II** (Non-dominated Sorting Genetic Algorithm II), reproducing the results of:

> Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002).
> *A fast and elitist multiobjective genetic algorithm: NSGA-II.*
> IEEE Transactions on Evolutionary Computation, 6(2), 182–197.

The algorithm is benchmarked against the paper's published numbers on the **ZDT1–ZDT4** and **ZDT6** test suites, using both convergence (Υ) and diversity (Δ) metrics.

---

## What NSGA-II does

NSGA-II is an elitist multi-objective evolutionary algorithm. Given a problem with two or more conflicting objectives, it evolves a population of candidate solutions toward the **Pareto-optimal front** — the set of solutions where no objective can be improved without worsening another.

Each generation:

1. **Generate offspring** from the current parent population using binary tournament selection, simulated binary crossover (SBX), and polynomial mutation.
2. **Combine** parents and offspring into a pool of size 2N.
3. **Fast non-dominated sort** the pool into Pareto fronts F₁, F₂, …
4. **Compute crowding distance** within each front (a density estimate that prefers solutions in less crowded regions of objective space).
5. **Select the top N** by the crowded-comparison operator: lower rank wins; ties broken by larger crowding distance.

This combination — non-dominated sorting for convergence + crowding distance for diversity — is what allows NSGA-II to find well-spread approximations of the true Pareto front in O(MN²) time per generation.

---

## Repository layout

```text
nsga-II/
├── individual.py     # Individual class
├── sorting.py        # Fast non-dominated sorting
├── crowding.py       # Crowding distance & crowded-comparison
├── operators.py      # Tournament selection, SBX, polynomial mutation
├── nsga2.py          # Main loop (init, evaluate, evolve, select)
├── problems.py       # ZDT1–ZDT4 definitions and true Pareto fronts
├── metrics.py        # Convergence (Υ) and diversity (Δ) metrics
├── main.py           # Orchestrates runs, statistics, and plotting
├── nsga_2.ipynb      # Same code packaged as a Jupyter / Colab notebook
└── results/          # Generated plots
```

### File-by-file

| File | Contents |
| ---- | -------- |
| [individual.py](individual.py) | `Individual` — holds the decision vector `x`, objective values, Pareto `rank`, and crowding `distance`. |
| [sorting.py](sorting.py) | `dominates(p, q)` — Pareto dominance check. `fast_nondominated_sort(population)` — partitions the population into 1-indexed Pareto fronts F₁, F₂, …, assigning a rank to each individual. |
| [crowding.py](crowding.py) | `crowding_distance_assignment(front)` — for each individual in a front, sums the normalised objective-space gap between its neighbours along every objective; boundary points get ∞. `crowded_comparison(a, b)` — the operator used by tournament selection and final truncation. |
| [operators.py](operators.py) | `tournament_selection` (binary tournament + crowded comparison), `sbx_crossover` (SBX with default `p_c=0.9`), `polynomial_mutation` (per-gene mutation with `p_m=1/n`). |
| [nsga2.py](nsga2.py) | `initialize_population`, `evaluate_population`, `make_new_population`, `nsga2_step` (one generation), and `run_nsga2` (full loop with seed control and history tracking). |
| [problems.py](problems.py) | The five ZDT benchmarks. Each entry in `PROBLEMS` provides the objective function, variable bounds, the true Pareto front (analytical), and `n_vars`. |
| [metrics.py](metrics.py) | `convergence_metric` (Υ) — mean min-distance from obtained solutions to a dense sample of the true front. `diversity_metric` (Δ) — non-uniformity of spacing along the obtained front, including boundary terms (Eq. 1 in the paper). |
| [main.py](main.py) | Driver: runs all four ZDTs once for plotting, then 10 independent trials per problem for statistical comparison against the paper's Table II / III values, then ZDT6 separately. Saves all plots to `results/`. |

---

## Test problems

| Problem | Front shape | n_vars | Bounds | What it stresses |
| ------- | ----------- | ------ | ------ | ---------------- |
| **ZDT1** | Convex | 30 | [0, 1]³⁰ | Baseline convergence on a simple convex front. |
| **ZDT2** | Non-convex (concave) | 30 | [0, 1]³⁰ | Algorithms biased to convex regions struggle here. |
| **ZDT3** | Disconnected (5 segments) | 30 | [0, 1]³⁰ | Spread across discontinuous regions. |
| **ZDT4** | Convex, 21⁹ local fronts | 10 | [0,1] × [-5,5]⁹ | Multimodality — easy to get trapped on local fronts. |
| **ZDT6** | Non-convex, biased density | 10 | [0, 1]¹⁰ | Non-uniform density along the front + thin feasible region near the front. |

---

## Running

Requires `numpy` and `matplotlib`. From the project directory:

```bash
python3 main.py
```

This produces all four plots in `results/` and prints per-problem metrics plus a side-by-side comparison against the paper's Tables II / III.

Or open [nsga_2.ipynb](nsga_2.ipynb) directly in Jupyter / Google Colab — it contains the full, self-contained pipeline.

### Default parameters (matching the paper)

| Parameter | Value |
| --------- | ----- |
| Population size `N` | 100 |
| Generations | 250 |
| SBX distribution index `η_c` | 20 |
| Mutation distribution index `η_m` | 20 |
| Crossover probability `p_c` | 0.9 |
| Mutation probability `p_m` | 1 / n_vars |
| Independent trials (statistics) | 10 |

---

## Outputs

All artefacts are written to `results/`:

| File | Description |
| ---- | ----------- |
| [results/pareto_fronts.png](results/pareto_fronts.png) | Final Front-1 vs. true Pareto front for ZDT1–ZDT4 (2×2 grid). |
| [results/convergence_curves.png](results/convergence_curves.png) | Best `f₁` per generation for each problem. |
| [results/metric_comparison.png](results/metric_comparison.png) | Bar chart: our mean Υ and Δ vs. the paper's Tables II / III, with std-dev error bars over 10 runs. |
| [results/zdt6_front.png](results/zdt6_front.png) | Pareto front for ZDT6. |

The console additionally prints per-trial Υ / Δ values and a side-by-side comparison against the published means and variances.

### Paper reference values (for the comparison plot)

These are the reported means / variances used by `main.py` to validate the implementation:

| Problem | Paper Υ (mean, var) | Paper Δ (mean, var) |
| ------- | ------------------- | ------------------- |
| ZDT1 | 0.033482, 0.004750 | 0.390307, 0.001876 |
| ZDT2 | 0.072391, 0.031689 | 0.430776, 0.004721 |
| ZDT3 | 0.114500, 0.007940 | 0.738540, 0.019706 |
| ZDT4 | 0.513053, 0.118460 | 0.702612, 0.064648 |
