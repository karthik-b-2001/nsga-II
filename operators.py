import numpy as np

from .crowding import crowded_comparison
from .individual import Individual


def tournament_selection(population):
    """Binary tournament: pick 2 at random, return crowded-comparison winner."""
    idx_a, idx_b = np.random.choice(len(population), size=2, replace=False)
    return crowded_comparison(population[idx_a], population[idx_b])


def sbx_crossover(parent1, parent2, eta_c, bounds, p_c=0.9):
    """Simulated Binary Crossover Returns two children."""
    n = len(parent1.x)
    child1_x = parent1.x.copy()
    child2_x = parent2.x.copy()

    if np.random.random() > p_c:
        return Individual(child1_x), Individual(child2_x)

    for i in range(n):
        lb, ub = bounds[i]
        if abs(parent1.x[i] - parent2.x[i]) < 1e-10:
            continue

        y1 = min(parent1.x[i], parent2.x[i])
        y2 = max(parent1.x[i], parent2.x[i])

        # Lower child
        u = np.random.random()
        beta = 1.0 + (2.0 * (y1 - lb) / (y2 - y1))
        alpha = 2.0 - beta ** (-(eta_c + 1.0))
        if u <= 1.0 / alpha:
            beta_q = (u * alpha) ** (1.0 / (eta_c + 1.0))
        else:
            beta_q = (1.0 / (2.0 - u * alpha)) ** (1.0 / (eta_c + 1.0))
        c1 = 0.5 * ((y1 + y2) - beta_q * (y2 - y1))

        # Upper child
        u = np.random.random()
        beta = 1.0 + (2.0 * (ub - y2) / (y2 - y1))
        alpha = 2.0 - beta ** (-(eta_c + 1.0))
        if u <= 1.0 / alpha:
            beta_q = (u * alpha) ** (1.0 / (eta_c + 1.0))
        else:
            beta_q = (1.0 / (2.0 - u * alpha)) ** (1.0 / (eta_c + 1.0))
        c2 = 0.5 * ((y1 + y2) + beta_q * (y2 - y1))

        c1 = float(np.clip(c1, lb, ub))
        c2 = float(np.clip(c2, lb, ub))

        if np.random.random() <= 0.5:
            child1_x[i], child2_x[i] = c1, c2
        else:
            child1_x[i], child2_x[i] = c2, c1

    return Individual(child1_x), Individual(child2_x)


def polynomial_mutation(individual, eta_m, bounds):
    """Polynomial mutation Mutates in place."""
    n = len(individual.x)
    p_m = 1.0 / n

    for i in range(n):
        if np.random.random() > p_m:
            continue
        lb, ub = bounds[i]
        y = individual.x[i]
        delta = ub - lb
        u = np.random.random()
        if u < 0.5:
            delta_q = (2.0 * u + (1.0 - 2.0 * u) *
                       (1.0 - (y - lb) / delta) ** (eta_m + 1.0)
                       ) ** (1.0 / (eta_m + 1.0)) - 1.0
        else:
            delta_q = 1.0 - (2.0 * (1.0 - u) + 2.0 * (u - 0.5) *
                             (1.0 - (ub - y) / delta) ** (eta_m + 1.0)
                             ) ** (1.0 / (eta_m + 1.0))
        individual.x[i] = float(np.clip(y + delta_q * delta, lb, ub))

    return individual
