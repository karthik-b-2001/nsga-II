import os
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import math
from copy import deepcopy

from individual import Individual
from sorting import dominates, fast_nondominated_sort
from crowding import crowding_distance_assignment, crowded_comparison
from operators import tournament_selection, sbx_crossover, polynomial_mutation
from nsga2 import (
    initialize_population,
    evaluate_population,
    make_new_population,
    nsga2_step,
    run_nsga2,
)
from problems import (
    zdt1, zdt1_bounds, zdt1_front,
    zdt2, zdt2_bounds, zdt2_front,
    zdt3, zdt3_bounds, zdt3_front,
    zdt4, zdt4_bounds, zdt4_front,
    PROBLEMS,
)
from metrics import convergence_metric, diversity_metric


plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.linewidth": 0.8,
    "figure.dpi": 120,
})

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results",
)
os.makedirs(RESULTS_DIR, exist_ok=True)


# =========================================================================
# Cell 17: Run NSGA-II on All ZDT Problems
# =========================================================================

# Paper parameters
N_POP = 100
N_GEN = 250
ETA_C = 20
ETA_M = 20
SEED  = 42

results = {}

for name in ["ZDT1", "ZDT2", "ZDT3", "ZDT4"]:
    print(f"\n{'='*50}")
    print(f"  Running NSGA-II on {name}")
    print(f"{'='*50}")

    cfg = PROBLEMS[name]
    bounds = cfg["bounds"](cfg["n_vars"])

    t0 = time.time()
    pop, history = run_nsga2(
        problem_fn=cfg["fn"], N=N_POP, n_vars=cfg["n_vars"],
        bounds=bounds, n_generations=N_GEN,
        eta_c=ETA_C, eta_m=ETA_M, seed=SEED
    )
    elapsed = time.time() - t0

    front1 = [ind for ind in pop if ind.rank == 1]
    f1 = [ind.objectives[0] for ind in front1]
    f2 = [ind.objectives[1] for ind in front1]

    results[name] = {"f1": f1, "f2": f2, "history": history, "front_size": len(front1)}
    print(f"  Done in {elapsed:.1f}s — Front-1 size: {len(front1)}")


# =========================================================================
# Cell 19: Pareto Front Plots
# =========================================================================

fig, axes = plt.subplots(2, 2, figsize=(11, 9))
axes = axes.flatten()

titles = [
    "ZDT1 (convex front)",
    "ZDT2 (non-convex front)",
    "ZDT3 (disconnected front)",
    "ZDT4 (many local fronts)",
]

for ax, name, title in zip(axes, ["ZDT1", "ZDT2", "ZDT3", "ZDT4"], titles):
    cfg = PROBLEMS[name]
    true_f1, true_f2 = cfg["front"](n_points=500)

    # True Pareto front
    ax.plot(true_f1, true_f2, "k-", linewidth=1.2, label="True Pareto front", zorder=1)

    # Obtained solutions
    r = results[name]
    order = np.argsort(r["f1"])
    ax.scatter(np.array(r["f1"])[order], np.array(r["f2"])[order],
               s=14, facecolors="none", edgecolors="royalblue",
               linewidths=0.8, zorder=2, label="NSGA-II obtained")

    ax.set_xlabel(r"$f_1$", fontsize=12)
    ax.set_ylabel(r"$f_2$", fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.15)

fig.suptitle("NSGA-II: Obtained vs True Pareto Fronts\n(N=100, 250 gen, ηc=20, ηm=20)",
             fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "pareto_fronts.png"), bbox_inches="tight", dpi=150)
plt.close()


# =========================================================================
# Cell 21: Convergence Curves
# =========================================================================

fig, axes = plt.subplots(2, 2, figsize=(11, 8))
axes = axes.flatten()

for ax, name in zip(axes, ["ZDT1", "ZDT2", "ZDT3", "ZDT4"]):
    h = results[name]["history"]
    ax.plot(range(1, len(h) + 1), h, color="steelblue", linewidth=1.5)
    ax.set_xlabel("Generation", fontsize=11)
    ax.set_ylabel("Best $f_1$", fontsize=11)
    ax.set_title(f"{name} — Convergence", fontsize=12)
    ax.grid(True, alpha=0.15)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "convergence_curves.png"), bbox_inches="tight", dpi=150)
plt.close()


# =========================================================================
# Cell 23: Single-Run Performance Metrics
# =========================================================================

ups_label = "Υ (conv.)"
dlt_label = "Δ (div.)"
print(f"{'Problem':<8} {'Front Size':>10} {ups_label:>14} {dlt_label:>14}")
print("-" * 48)

for name in ["ZDT1", "ZDT2", "ZDT3", "ZDT4"]:
    r = results[name]
    cfg = PROBLEMS[name]
    true_f1, true_f2 = cfg["front"](n_points=500)

    ups = convergence_metric(r["f1"], r["f2"], true_f1, true_f2)
    dlt = diversity_metric(r["f1"], r["f2"], true_f1, true_f2)

    print(f"{name:<8} {r['front_size']:>10} {ups:>14.6f} {dlt:>14.6f}")


# =========================================================================
# Cell 25: Statistical Validation (10 Independent Trials)
# =========================================================================

# Published values from Tables II and III (real-coded NSGA-II with real parameters)
# Each tuple: (mean_upsilon, var_upsilon, mean_delta, var_delta)
PAPER_VALUES = {
    "ZDT1": (0.033482, 0.004750, 0.390307, 0.001876),
    "ZDT2": (0.072391, 0.031689, 0.430776, 0.004721),
    "ZDT3": (0.114500, 0.007940, 0.738540, 0.019706),
    "ZDT4": (0.513053, 0.118460, 0.702612, 0.064648),
}

N_RUNS = 10
trial_results = {}

for name in ["ZDT1", "ZDT2", "ZDT3", "ZDT4"]:
    cfg = PROBLEMS[name]
    bounds = cfg["bounds"](cfg["n_vars"])
    true_f1, true_f2 = cfg["front"](n_points=500)

    upsilons, deltas = [], []
    print(f"\n{'='*60}")
    print(f"  {name} — {N_RUNS} independent runs")
    print(f"{'='*60}")

    for run in range(N_RUNS):
        pop, _ = run_nsga2(
            problem_fn=cfg["fn"], N=N_POP, n_vars=cfg["n_vars"],
            bounds=bounds, n_generations=N_GEN,
            eta_c=ETA_C, eta_m=ETA_M, seed=100 + run,
        )
        front1 = [ind for ind in pop if ind.rank == 1]
        f1 = [ind.objectives[0] for ind in front1]
        f2 = [ind.objectives[1] for ind in front1]
        ups = convergence_metric(f1, f2, true_f1, true_f2)
        dlt = diversity_metric(f1, f2, true_f1, true_f2)
        upsilons.append(ups)
        deltas.append(dlt)
        print(f"  Run {run+1:2d}: Υ = {ups:.6f}   Δ = {dlt:.6f}")

    trial_results[name] = (upsilons, deltas)

print("\nAll trials complete.")


# =========================================================================
# Cell 27: Comparison with Paper
# =========================================================================

# ---- Convergence metric Y — Table II comparison ----
print("=" * 80)
print("CONVERGENCE METRIC Υ — Comparison with Table II (Deb et al. 2002)")
print("=" * 80)
print(f"{'Problem':<8} {'Our Mean':>14} {'Our Var':>14} {'Paper Mean':>14} {'Paper Var':>14}")
print("-" * 80)
for name, (upsilons, _) in trial_results.items():
    our_mean = np.mean(upsilons)
    our_var  = np.var(upsilons)
    p_mean, p_var, _, _ = PAPER_VALUES[name]
    print(f"{name:<8} {our_mean:>14.6f} {our_var:>14.6f} {p_mean:>14.6f} {p_var:>14.6f}")

print()
# ---- Diversity metric D — Table III comparison ----
print("=" * 80)
print("DIVERSITY METRIC Δ — Comparison with Table III (Deb et al. 2002)")
print("=" * 80)
print(f"{'Problem':<8} {'Our Mean':>14} {'Our Var':>14} {'Paper Mean':>14} {'Paper Var':>14}")
print("-" * 80)
for name, (_, deltas) in trial_results.items():
    our_mean = np.mean(deltas)
    our_var  = np.var(deltas)
    _, _, p_mean, p_var = PAPER_VALUES[name]
    print(f"{name:<8} {our_mean:>14.6f} {our_var:>14.6f} {p_mean:>14.6f} {p_var:>14.6f}")


# =========================================================================
# Cell 29: Visual Comparison using Bar Charts
# =========================================================================

problems = ["ZDT1", "ZDT2", "ZDT3", "ZDT4"]
x_pos = np.arange(len(problems))
width = 0.35

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# ---- Convergence metric Υ ----
our_ups_means = [np.mean(trial_results[p][0]) for p in problems]
our_ups_stds  = [np.std(trial_results[p][0]) for p in problems]
paper_ups_means = [PAPER_VALUES[p][0] for p in problems]

ax1.bar(x_pos - width/2, our_ups_means, width, yerr=our_ups_stds,
        label="Our Implementation", color="steelblue", capsize=4)
ax1.bar(x_pos + width/2, paper_ups_means, width,
        label="Paper (Table II)", color="coral", capsize=4)
ax1.set_xlabel("Problem")
ax1.set_ylabel("Mean Υ")
ax1.set_title("Convergence Metric Υ (lower is better)")
ax1.set_xticks(x_pos)
ax1.set_xticklabels(problems)
ax1.legend()
ax1.grid(True, alpha=0.15, axis="y")

# ---- Diversity metric Δ ----
our_dlt_means = [np.mean(trial_results[p][1]) for p in problems]
our_dlt_stds  = [np.std(trial_results[p][1]) for p in problems]
paper_dlt_means = [PAPER_VALUES[p][2] for p in problems]

ax2.bar(x_pos - width/2, our_dlt_means, width, yerr=our_dlt_stds,
        label="Our Implementation", color="steelblue", capsize=4)
ax2.bar(x_pos + width/2, paper_dlt_means, width,
        label="Paper (Table III)", color="coral", capsize=4)
ax2.set_xlabel("Problem")
ax2.set_ylabel("Mean Δ")
ax2.set_title("Diversity Metric Δ (lower is better)")
ax2.set_xticks(x_pos)
ax2.set_xticklabels(problems)
ax2.legend()
ax2.grid(True, alpha=0.15, axis="y")

plt.suptitle("Our NSGA-II vs Paper Results (10 independent runs each)", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "metric_comparison.png"), bbox_inches="tight", dpi=150)
plt.close()


# =========================================================================
# Cell 31: ZDT6 (Nonuniform Pareto-Optimal Density)
# =========================================================================

# ---- ZDT6 definition ----
def zdt6(x):
    n = len(x)
    f1 = 1.0 - np.exp(-4.0 * x[0]) * (np.sin(6.0 * np.pi * x[0])) ** 6
    g  = 1.0 + 9.0 * (np.sum(x[1:]) / (n - 1)) ** 0.25
    f2 = g * (1.0 - (f1 / g) ** 2)
    return [f1, f2]

def zdt6_bounds(n=10):
    return [(0.0, 1.0)] * n

def zdt6_front(n_points=500):
    # Pareto front: f2 = 1 - f1^2 for f1 in [0.2807753191, 1]
    f1 = np.linspace(0.2807753191, 1.0, n_points)
    return f1, 1.0 - f1 ** 2

PROBLEMS["ZDT6"] = {"fn": zdt6, "bounds": zdt6_bounds, "front": zdt6_front, "n_vars": 10}

# ---- Run NSGA-II on ZDT6 ----
print("Running NSGA-II on ZDT6 ...")
cfg = PROBLEMS["ZDT6"]
bounds = cfg["bounds"](cfg["n_vars"])
pop_zdt6, hist_zdt6 = run_nsga2(
    problem_fn=cfg["fn"], N=N_POP, n_vars=cfg["n_vars"],
    bounds=bounds, n_generations=N_GEN,
    eta_c=ETA_C, eta_m=ETA_M, seed=SEED,
)
front1 = [ind for ind in pop_zdt6 if ind.rank == 1]
f1_zdt6 = [ind.objectives[0] for ind in front1]
f2_zdt6 = [ind.objectives[1] for ind in front1]

# ---- Plot ----
fig, ax = plt.subplots(1, 1, figsize=(6, 5))
true_f1, true_f2 = zdt6_front(500)
ax.plot(true_f1, true_f2, "k-", linewidth=1.2, label="True Pareto front", zorder=1)
order = np.argsort(f1_zdt6)
ax.scatter(np.array(f1_zdt6)[order], np.array(f2_zdt6)[order],
           s=14, facecolors="none", edgecolors="royalblue",
           linewidths=0.8, zorder=2, label="NSGA-II obtained")
ax.set_xlabel(r"$f_1$"); ax.set_ylabel(r"$f_2$")
ax.set_title("ZDT6 — Pareto Front")
ax.legend(); ax.grid(True, alpha=0.15)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "zdt6_front.png"), bbox_inches="tight", dpi=150)
plt.close()

# ---- Metrics ----
ups = convergence_metric(f1_zdt6, f2_zdt6, true_f1, true_f2)
dlt = diversity_metric(f1_zdt6, f2_zdt6, true_f1, true_f2)
print(f"\nZDT6 front size: {len(front1)}")
print(f"ZDT6 Υ = {ups:.6f}   Δ = {dlt:.6f}")

print(f"\nAll plots written to {RESULTS_DIR}")
