# GNN-Guided A* For Shortest Distance

This workbook integrates a Graph Neural Network (GNN) as a heuristic function into the A* algorithm, aiming to reduce the number of explored nodes and increase search efficiency.

Traditional heuristic functions in A* like Euclidean or Manhattan distance cannot avoid exploring dead ends, which results in wasted computation time. While some machine learning methods can predict the distance from candidate nodes to the goal, they typically require extremely high training costs.

For example, even with a small graph of just 100 nodes, you would need to consider 4,950 node pairs (i.e., Combination(100, 2)) for training. In real-world scenarios, graphs with millions of nodes are common, making it impractical to prepare exhaustive training datasets.

In this notebook, by selecting a small number of representative nodes, the GNN can still perform effectively by leveraging its generalization ability, significantly improving search efficiency without the need for massive training data.

---

This repository implements a grid-based pathfinding system augmented by Graph Neural Network (GNN) heuristics. It includes:

- **Traditional GNN** (`gnn.py`)
- **GraphSAGE** variant (`sageconv.py`)
- **GCN** variant (`gcn.py`)
- **Utility functions** (`utils.py`)
- Graph construction and data generation utilities (`create_graph.py`, `create_data.py`)
- A* benchmark implementation (`astar.py`)
- Visualization tools (`visualize.py`)
- A Jupyter notebook driver (`main.ipynb`)

---

## Directory Structure
```
├── create_graph.py           # Builds grid graph with obstacles
├── create_data.py            # Generates training samples from landmark pairs
├── astar.py                  # Traditional A* for shortest distance
├── gnn_guided_astar.py       # Traditional A* integrated with trained GNN for shortest distance
├── main.ipynb                # Notebook: data prep, training, evaluation, visualization
├── graph_neural_network/     # GNN implementations and utils
│   ├── __init__.py
│   ├── gnn.py                # Traditional GNN model
│   ├── sageconv.py           # GraphSAGE-based model
│   ├── gcn.py                # GCN-based model
│   └── utils.py              # Shared evaluation and RMSE functions
└── README.md                 # This file
```

## Requirements

- Python 3.8+
- PyTorch
- torch-geometric
- NetworkX
- NumPy
- Matplotlib

Install via pip:
```bash
pip install torch torchvision torch-geometric networkx numpy matplotlib
```

## Usage

1. Launch the Jupyter notebook:
   ```bash
   jupyter lab main.ipynb
   ```
2. Run cells in order to:
   - Build the grid graph and generate training data
   - Train each GNN variant (`gnn.py`, `sageconv.py`, `gcn.py`)
   - Evaluate heuristics against A*
   - Visualize learned heuristic fields and path exploration

## Contributing

- Add new GNN variants in `graph_neural_network/`
- Extend visualization in `visualize.py`
- Report issues or request features via GitHub pull requests

---

*Created on July 29, 2025*
