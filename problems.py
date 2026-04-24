import numpy as np


# ---- ZDT1: Convex front ----
def zdt1(x):
    f1 = x[0]
    g = 1.0 + 9.0 * np.sum(x[1:]) / (len(x) - 1)
    f2 = g * (1.0 - np.sqrt(f1 / g))
    return [f1, f2]

def zdt1_bounds(n=30): return [(0.0, 1.0)] * n
def zdt1_front(n_points=500):
    f1 = np.linspace(0, 1, n_points)
    return f1, 1.0 - np.sqrt(f1)

# ---- ZDT2: Non-convex front ----
def zdt2(x):
    f1 = x[0]
    g = 1.0 + 9.0 * np.sum(x[1:]) / (len(x) - 1)
    f2 = g * (1.0 - (f1 / g) ** 2)
    return [f1, f2]

def zdt2_bounds(n=30): return [(0.0, 1.0)] * n
def zdt2_front(n_points=500):
    f1 = np.linspace(0, 1, n_points)
    return f1, 1.0 - f1 ** 2

# ---- ZDT3: Disconnected front ----
def zdt3(x):
    f1 = x[0]
    g = 1.0 + 9.0 * np.sum(x[1:]) / (len(x) - 1)
    f2 = g * (1.0 - np.sqrt(f1 / g) - (f1 / g) * np.sin(10.0 * np.pi * f1))
    return [f1, f2]

def zdt3_bounds(n=30): return [(0.0, 1.0)] * n
def zdt3_front(n_points=1000):
    f1 = np.linspace(0, 1, n_points)
    f2 = 1.0 - np.sqrt(f1) - f1 * np.sin(10.0 * np.pi * f1)
    # Keep only non-dominated points
    pf1, pf2 = [], []
    for i in range(len(f1)):
        dominated = False
        for j in range(len(f1)):
            if i != j and f1[j] <= f1[i] and f2[j] <= f2[i] and (f1[j] < f1[i] or f2[j] < f2[i]):
                dominated = True
                break
        if not dominated:
            pf1.append(f1[i]); pf2.append(f2[i])
    return np.array(pf1), np.array(pf2)

# ---- ZDT4: Many local fronts ----
def zdt4(x):
    f1 = x[0]
    g = 1.0 + 10.0 * (len(x) - 1) + np.sum(x[1:]**2 - 10.0 * np.cos(4.0 * np.pi * x[1:]))
    f2 = g * (1.0 - np.sqrt(f1 / g))
    return [f1, f2]

def zdt4_bounds(n=10):
    return [(0.0, 1.0)] + [(-5.0, 5.0)] * (n - 1)
def zdt4_front(n_points=500):
    f1 = np.linspace(0, 1, n_points)
    return f1, 1.0 - np.sqrt(f1)

# ---- Problems ----
PROBLEMS = {
    "ZDT1": {"fn": zdt1, "bounds": zdt1_bounds, "front": zdt1_front, "n_vars": 30},
    "ZDT2": {"fn": zdt2, "bounds": zdt2_bounds, "front": zdt2_front, "n_vars": 30},
    "ZDT3": {"fn": zdt3, "bounds": zdt3_bounds, "front": zdt3_front, "n_vars": 30},
    "ZDT4": {"fn": zdt4, "bounds": zdt4_bounds, "front": zdt4_front, "n_vars": 10},
}
