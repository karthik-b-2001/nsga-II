import numpy as np

from .crowding import crowding_distance_assignment
from .individual import Individual
from .operators import polynomial_mutation, sbx_crossover, tournament_selection
from .sorting import fast_nondominated_sort


def initialize_population(N, n_vars, bounds):
    """Create N individuals with uniformly random decision variables."""
    return [Individual([np.random.uniform(lb, ub) for lb, ub in bounds]) for _ in range(N)]


def evaluate_population(population, problem_fn):
    """Evaluate objective values for every individual."""
    for ind in population:
        ind.objectives = problem_fn(ind.x)


def make_new_population(population, bounds, eta_c=20, eta_m=20):
    """Generate N offspring via tournament selection + SBX + polynomial mutation."""
    N = len(population)
    offspring = []
    while len(offspring) < N:
        p1 = tournament_selection(population)
        p2 = tournament_selection(population)
        c1, c2 = sbx_crossover(p1, p2, eta_c, bounds)
        polynomial_mutation(c1, eta_m, bounds)
        polynomial_mutation(c2, eta_m, bounds)
        offspring.append(c1)
        if len(offspring) < N:
            offspring.append(c2)
    return offspring


def nsga2_step(parents, offspring, N):
    """One generation: combine 2N, sort, select best N by rank + crowding."""
    combined = parents + offspring
    fronts = fast_nondominated_sort(combined)
    next_pop = []
    for front in fronts:
        crowding_distance_assignment(front)
        if len(next_pop) + len(front) <= N:
            next_pop.extend(front)
        else:
            front.sort(key=lambda ind: ind.distance, reverse=True)
            next_pop.extend(front[:N - len(next_pop)])
            break
    return next_pop


def run_nsga2(problem_fn, N, n_vars, bounds, n_generations,
              eta_c=20, eta_m=20, seed=None):
    """Run the full NSGA-II algorithm. Returns (final_population, history)."""
    if seed is not None:
        np.random.seed(seed)

    population = initialize_population(N, n_vars, bounds)
    evaluate_population(population, problem_fn)
    fronts = fast_nondominated_sort(population)
    for front in fronts:
        crowding_distance_assignment(front)

    history = []
    for gen in range(n_generations):
        offspring = make_new_population(population, bounds, eta_c, eta_m)
        evaluate_population(offspring, problem_fn)
        population = nsga2_step(population, offspring, N)
        best_f1 = min(ind.objectives[0] for ind in population)
        history.append(best_f1)
        if (gen + 1) % 50 == 0:
            print(f"  Gen {gen+1}/{n_generations} — best f1: {best_f1:.6f}")

    return population, history
