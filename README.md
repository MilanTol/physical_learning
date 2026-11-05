# physical-learning

Coupled learning (Stern et al.) in disordered flow networks.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Run
```bash
python scripts/demo.py     # writes results/demo.png
pytest                     # gradient-equivalence test
```

## Layout
- `network.py`  : `FlowNetwork` (graph, Laplacian, constrained steady-state solve)
- `coupled_learning.py` : `Task`, `cost`, `coupled_learning_step`, `train`, `check_gradient`
- `plotting.py` : network / training plots
