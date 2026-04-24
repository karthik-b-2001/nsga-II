import numpy as np


class Individual:
    def __init__(self, x):
        self.x = np.array(x, dtype=float)
        self.objectives = []
        self.rank = None
        self.distance = 0.0

    def __repr__(self):
        return (f"Individual(rank={self.rank}, dist={self.distance:.3f}, "
                f"obj={[round(o, 4) for o in self.objectives]})")
