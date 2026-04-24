import numpy as np


def convergence_metric(obtained_f1, obtained_f2, true_f1, true_f2):
    """Average min Euclidean distance from obtained solutions to true front."""
    obtained = np.column_stack([obtained_f1, obtained_f2])
    reference = np.column_stack([true_f1, true_f2])
    min_dists = []
    for sol in obtained:
        dists = np.sqrt(np.sum((reference - sol) ** 2, axis=1))
        min_dists.append(np.min(dists))
    return float(np.mean(min_dists))


def diversity_metric(obtained_f1, obtained_f2, true_f1, true_f2):
    """Non-uniformity measure Δ from Eq. (1) of the paper."""
    order = np.argsort(obtained_f1)
    f1 = np.array(obtained_f1)[order]
    f2 = np.array(obtained_f2)[order]
    N = len(f1)
    if N < 2:
        return float('inf')

    d = np.sqrt(np.diff(f1)**2 + np.diff(f2)**2)
    d_mean = np.mean(d)

    # Boundary distances to extreme points on true front
    d_f = float(np.sqrt((f1[0] - true_f1[0])**2 + (f2[0] - true_f2[0])**2))
    d_l = float(np.sqrt((f1[-1] - true_f1[-1])**2 + (f2[-1] - true_f2[-1])**2))

    numerator = d_f + d_l + np.sum(np.abs(d - d_mean))
    denominator = d_f + d_l + (N - 1) * d_mean
    return 0.0 if denominator == 0 else float(numerator / denominator)
