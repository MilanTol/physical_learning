import numpy as np
from physical_learning import FlowNetwork, Task, check_gradient, train, cost


def _setup():
    net = FlowNetwork.random_delaunay(n_side=6, seed=0)
    n = np.random.default_rng(1).permutation(net.N)
    return net, Task(n[:2], [1.0, 0.0], n[2:3], [0.8])


def test_cl_matches_gradient():
    net, task = _setup()
    cos, dk, g = check_gradient(net, task)
    assert cos > 0.999
    # CL with this normalization descends sum (p_T - P_T)^2, i.e. dk -> 2 * (-dC/dk) for C = 1/2 sum(...)^2
    assert np.linalg.norm(dk - 2 * g) / np.linalg.norm(2 * g) < 0.05


def test_training_reduces_cost():
    net, task = _setup()
    c0 = cost(net, task)
    train(net, [task], n_epochs=100, alpha=0.5, k_min=1e-2, k_max=20.0, verbose=False)
    assert cost(net, task) < 1e-2 * c0
