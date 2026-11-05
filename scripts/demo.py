import matplotlib
matplotlib.use("Agg")  # headless; avoids the GTK backend crash
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from physical_learning import FlowNetwork, Task, cost, train, check_gradient
from physical_learning.plotting import plot_network

net = FlowNetwork.random_delaunay(n_side=10, seed=1)
rng = np.random.default_rng(2)
nodes = rng.permutation(net.N)
task = Task(nodes[:2], [1.0, 0.0], nodes[2:3], [0.8])

k_init = net.k.copy()
print("gradient cos:", check_gradient(net, task)[0])
hist = train(net, [task], n_epochs=300, eta=1e-3, alpha=0.5, k_min=1e-2, k_max=20.0)

fig, ax = plt.subplots(1, 3, figsize=(14, 4))
plot_network(net, task, ax[0], k=k_init, title="before")
plot_network(net, task, ax[1], title="after")
ax[2].semilogy(hist["cost"]); ax[2].set_xlabel("epoch"); ax[2].set_ylabel("cost")
Path("results").mkdir(exist_ok=True)
plt.tight_layout(); plt.savefig("results/demo.png", dpi=150)
