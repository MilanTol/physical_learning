"""Flow network: graph, Laplacian, steady state under pressure constraints."""
from __future__ import annotations
import numpy as np
from scipy.spatial import Delaunay


class FlowNetwork:
    def __init__(self, edges: np.ndarray, pos: np.ndarray, k0: np.ndarray):
        self.edges = edges            # (E, 2) int
        self.pos = pos                # (N, 2), plotting only
        self.N = len(pos)
        self.E = len(edges)
        self.k = k0.astype(float).copy()
        B = np.zeros((self.E, self.N))   # dp = B @ p
        B[np.arange(self.E), edges[:, 0]] = 1.0
        B[np.arange(self.E), edges[:, 1]] = -1.0
        self.B = B

    @classmethod
    def random_delaunay(cls, n_side=10, jitter=0.35, k_range=(0.5, 1.5), seed=0):
        rng = np.random.default_rng(seed)
        g = np.stack(np.meshgrid(np.arange(n_side), np.arange(n_side)), -1).reshape(-1, 2).astype(float)
        pos = g + rng.uniform(-jitter, jitter, g.shape)
        tri = Delaunay(pos)
        e = set()
        for s in tri.simplices:
            for a, b in ((0, 1), (1, 2), (0, 2)):
                i, j = sorted((int(s[a]), int(s[b])))
                e.add((i, j))
        edges = np.array(sorted(e))
        return cls(edges, pos, rng.uniform(*k_range, len(edges)))

    def laplacian(self, k=None):
        k = self.k if k is None else k
        return self.B.T @ (k[:, None] * self.B)

    def solve(self, constrained_idx, constrained_val, k=None):
        """Minimize P over unconstrained nodes. Returns full pressure vector."""
        L = self.laplacian(k)
        c = np.asarray(constrained_idx)
        f = np.setdiff1d(np.arange(self.N), c)
        p = np.zeros(self.N)
        p[c] = constrained_val
        p[f] = np.linalg.solve(L[np.ix_(f, f)], -L[np.ix_(f, c)] @ p[c])
        return p

    def power(self, p, k=None):
        k = self.k if k is None else k
        return 0.5 * np.sum(k * (self.B @ p) ** 2)
