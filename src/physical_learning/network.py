"""Flow network: graph, Laplacian, steady state under pressure constraints.

Conventions
-----------
N nodes carry pressures p (shape (N,)); E edges ("pipes") carry conductances k (shape (E,)).
Pressure drop on edge j = (i, l) is  dp_j = p_i - p_l.  Stacked over all edges:  dp = B @ p.
Dissipated power:  P(p; k) = 1/2 * sum_j k_j * dp_j**2 = 1/2 * p^T L p,   with L = B^T diag(k) B.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial import Delaunay


class FlowNetwork:
    def __init__(self, edges: np.ndarray, pos: np.ndarray, k0: np.ndarray):
        """
        edges : (E, 2) int array, node index pairs (i, l) for each pipe
        pos   : (N, 2) float array, node coordinates (used for plotting only, not physics)
        k0    : (E,) initial conductances (the learning degrees of freedom)
        """
        self.edges = edges
        self.pos = pos
        self.N = len(pos)             # number of nodes
        self.E = len(edges)           # number of pipes
        self.k = k0.astype(float).copy()   # copy so the caller's array is never mutated

        # Edge-node incidence matrix B (E x N): row j has +1 at the first node and -1 at
        # the second node of edge j, so (B @ p)[j] = p_i - p_l = pressure drop across pipe j.
        # Orientation is arbitrary: dp enters the physics only squared.
        B = np.zeros((self.E, self.N))
        B[np.arange(self.E), edges[:, 0]] = 1.0
        B[np.arange(self.E), edges[:, 1]] = -1.0
        self.B = B

    @classmethod
    def random_delaunay(cls, n_side=10, jitter=0.35, k_range=(0.5, 1.5), seed=0):
        """Disordered 2D network: jittered n_side x n_side grid, connected by Delaunay triangulation.

        jitter  : max displacement of each node from its lattice site (0 = regular triangulated grid)
        k_range : initial conductances drawn uniformly from this interval
        seed    : RNG seed, for reproducible networks
        """
        rng = np.random.default_rng(seed)

        # Regular grid coordinates, shape (n_side**2, 2)
        g = np.stack(np.meshgrid(np.arange(n_side), np.arange(n_side)), -1).reshape(-1, 2).astype(float)
        # Random displacement of every node -> positional disorder
        pos = g + rng.uniform(-jitter, jitter, g.shape)

        # Triangulate; each triangle contributes its 3 sides as edges
        tri = Delaunay(pos)
        e = set()   # set removes duplicates: interior edges are shared by two triangles
        for s in tri.simplices:
            for a, b in ((0, 1), (1, 2), (0, 2)):
                # sort so (i, l) and (l, i) map to the same edge
                i, j = sorted((int(s[a]), int(s[b])))
                e.add((i, j))
        edges = np.array(sorted(e))   # sorted -> deterministic edge ordering

        return cls(edges, pos, rng.uniform(*k_range, len(edges)))

    def laplacian(self, k=None):
        """Weighted graph Laplacian L = B^T diag(k) B, shape (N, N).

        Symmetric, rows sum to zero (uniform pressure costs no power), positive semi-definite.
        k : optional conductances to use instead of self.k (e.g. for finite-difference checks).
        """
        k = self.k if k is None else k
        # k[:, None] * B scales row j of B by k_j, equivalent to diag(k) @ B but without building diag(k)
        return self.B.T @ (k[:, None] * self.B)

    def solve(self, constrained_idx, constrained_val, k=None):
        """Steady state: minimize P over the unconstrained nodes. Returns full pressure vector (N,).

        constrained_idx : node indices whose pressures are imposed
                          (free state: sources; clamped state: sources + targets)
        constrained_val : imposed pressures at those nodes, same length
        k               : optional conductances override

        Setting dP/dp_f = 0 and splitting L into free (f) and constrained (c) blocks:
            L_ff p_f + L_fc p_c = 0   ->   p_f = -L_ff^{-1} L_fc p_c
        This is Kirchhoff's current law (zero net flow) at every unconstrained node.
        """
        L = self.laplacian(k)
        c = np.asarray(constrained_idx)
        f = np.setdiff1d(np.arange(self.N), c)    # all remaining nodes (targets/hidden in free state)

        p = np.zeros(self.N)
        p[c] = constrained_val                    # impose the constraints
        # L_ff is invertible as long as every connected component contains a constrained node
        # and all k > 0 (hence the k_min clip in the learning step)
        p[f] = np.linalg.solve(L[np.ix_(f, f)], -L[np.ix_(f, c)] @ p[c])
        return p

    def power(self, p, k=None):
        """Dissipated power P = 1/2 * sum_j k_j * dp_j^2 for a given pressure vector p."""
        k = self.k if k is None else k
        return 0.5 * np.sum(k * (self.B @ p) ** 2)