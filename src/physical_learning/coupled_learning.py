"""
Coupled learning in a disordered flow network.

Physics:   P(p; k) = 1/2 sum_j k_j (dp_j)^2
Learning:  dk_j/dt = (alpha/eta) * 1/2 * [ (dp_j^F)^2 - (dp_j^C)^2 ]
Nudge:     p_T^C = p_T^F + eta * (P_T - p_T^F)
"""
from __future__ import annotations
import numpy as np
from .network import FlowNetwork


class Task:
    """One input/output pair: source pressures -> desired target pressures."""
    def __init__(self, src_idx, src_val, tgt_idx, tgt_val):
        self.S = np.asarray(src_idx)
        self.PS = np.asarray(src_val, float)
        self.T = np.asarray(tgt_idx)
        self.PT = np.asarray(tgt_val, float)


def cost(net: FlowNetwork, task: Task) -> float:
    pF = net.solve(task.S, task.PS)
    return 0.5 * np.sum((pF[task.T] - task.PT) ** 2)


def coupled_learning_step(net, task, eta=1e-3, alpha=1e-1, k_min=1e-3, k_max=None, apply=True):
    pF = net.solve(task.S, task.PS)
    pT_C = pF[task.T] + eta * (task.PT - pF[task.T])
    pC = net.solve(np.concatenate([task.S, task.T]), np.concatenate([task.PS, pT_C]))
    dpF, dpC = net.B @ pF, net.B @ pC
    dk = 0.5 * alpha / eta * (dpF**2 - dpC**2)
    if apply:
        net.k = np.clip(net.k + dk, k_min, k_max)
    return dk


def train(net, tasks, n_epochs=500, eta=1e-3, alpha=1e-1, k_min=1e-3, k_max=None,
          sequential=True, verbose=True):
    hist = {"cost": [], "k": []}
    for ep in range(n_epochs):
        if sequential:
            for t in tasks:
                coupled_learning_step(net, t, eta, alpha, k_min, k_max)
        else:
            dk = sum(coupled_learning_step(net, t, eta, alpha, apply=False) for t in tasks) / len(tasks)
            net.k = np.clip(net.k + dk, k_min, k_max)
        c = np.mean([cost(net, t) for t in tasks])
        hist["cost"].append(c)
        hist["k"].append(net.k.copy())
        if verbose and (ep % max(1, n_epochs // 10) == 0 or ep == n_epochs - 1):
            print(f"epoch {ep:5d}  cost {c:.3e}")
    hist["k"] = np.array(hist["k"])
    return hist


def check_gradient(net, task, eta=1e-5, h=1e-6):
    """Cosine between coupled-learning dk and -dC/dk (finite differences), C = 1/2 sum (p_T - P_T)^2.
    With alpha=1, dk -> 2 * (-dC/dk) as eta -> 0 (CL descends sum (p_T - P_T)^2)."""
    dk = coupled_learning_step(net, task, eta=eta, alpha=1.0, apply=False)
    g = np.zeros(net.E)
    for j in range(net.E):
        kp, km = net.k.copy(), net.k.copy()
        kp[j] += h
        km[j] -= h
        cp = 0.5 * np.sum((net.solve(task.S, task.PS, kp)[task.T] - task.PT) ** 2)
        cm = 0.5 * np.sum((net.solve(task.S, task.PS, km)[task.T] - task.PT) ** 2)
        g[j] = (cp - cm) / (2 * h)
    cos = dk @ (-g) / (np.linalg.norm(dk) * np.linalg.norm(g))
    return cos, dk, -g
