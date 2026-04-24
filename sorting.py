def dominates(p, q):
    """p dominates q if p is no worse on ALL objectives and strictly better on at least one."""
    at_least_one_better = False
    for p_obj, q_obj in zip(p.objectives, q.objectives):
        if p_obj > q_obj:
            return False
        if p_obj < q_obj:
            at_least_one_better = True
    return at_least_one_better


def fast_nondominated_sort(population):
    """Sort population into non-dominated fronts F1, F2, ... (1-indexed ranks)."""
    n = len(population)
    dominated_by_p = [[] for _ in range(n)]  # S_p
    domination_count = [0] * n                # n_p
    fronts = [[]]

    for i, p in enumerate(population):
        for j, q in enumerate(population):
            if i == j:
                continue
            if dominates(p, q):
                dominated_by_p[i].append(j)
            elif dominates(q, p):
                domination_count[i] += 1
        if domination_count[i] == 0:
            p.rank = 1
            fronts[0].append(i)

    current_front = 0
    while fronts[current_front]:
        next_front = []
        for i in fronts[current_front]:
            for j in dominated_by_p[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    population[j].rank = current_front + 2
                    next_front.append(j)
        current_front += 1
        fronts.append(next_front)

    fronts = [f for f in fronts if f]
    return [[population[i] for i in front] for front in fronts]
