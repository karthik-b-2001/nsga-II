import math


def crowding_distance_assignment(front):
    """Assign crowding distance to every individual in a single front."""
    n = len(front)
    if n == 0:
        return
    for ind in front:
        ind.distance = 0.0

    num_objectives = len(front[0].objectives)
    for m in range(num_objectives):
        front.sort(key=lambda ind: ind.objectives[m])
        f_min = front[0].objectives[m]
        f_max = front[-1].objectives[m]
        front[0].distance = math.inf
        front[-1].distance = math.inf
        if f_max == f_min:
            continue
        for i in range(1, n - 1):
            front[i].distance += (
                (front[i + 1].objectives[m] - front[i - 1].objectives[m])
                / (f_max - f_min)
            )


def crowded_comparison(ind_a, ind_b):
    """Crowded comparison operator: lower rank wins; ties broken by larger distance."""
    if ind_a.rank < ind_b.rank:
        return ind_a
    if ind_b.rank < ind_a.rank:
        return ind_b
    return ind_a if ind_a.distance > ind_b.distance else ind_b
