import matplotlib.pyplot as plt


def plot_network(net, task=None, ax=None, k=None, title=""):
    ax = ax or plt.gca()
    k = net.k if k is None else k
    for (i, j), kj in zip(net.edges, k):
        ax.plot(*net.pos[[i, j]].T, "k-", lw=0.3 + 3.0 * kj / k.max(), alpha=0.7)
    if task is not None:
        ax.scatter(*net.pos[task.S].T, c="tab:red", s=60, zorder=3, label="source")
        ax.scatter(*net.pos[task.T].T, c="tab:blue", s=60, zorder=3, label="target")
        ax.legend(loc="upper right", fontsize=7)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title)
