import numpy as np
import networkx as nx
import math
from create_data import get_edge_index
import torch

def get_heuristic(model, start, goal, G, mapping, edge_index, obstacles):
    DIV = 11
    x = torch.zeros(len(mapping), 6)
    for n, idx in mapping.items():
        dx, dy = n[0] - goal[0], n[1] - goal[1]
        r = math.hypot(dx, dy)
        theta = math.atan2(dy, dx) / math.pi
        x[idx] = torch.tensor([dx/DIV, dy/DIV, r/DIV, theta,
                               float(n not in obstacles), float(n == goal)])
    model.eval()
    with torch.no_grad():
        pred = model(x, edge_index)
    return pred[mapping[start]].item(), pred

# ===== Utility =====
def compute_rmse(true_list, pred_list):
    arr_t = np.array(true_list, dtype=float)
    arr_p = np.array(pred_list, dtype=float)
    return float(math.sqrt(np.mean((arr_t - arr_p)**2)))

# ===== Evaluation =====
def evaluate_model(model, valid_pairs, G, mapping, obstacles):
    print("\n=== Evaluation per Landmark Pair ===")
    errors, true_dists, pred_dists = [], [], []
    eidx = get_edge_index(G, mapping)

    for s, g in valid_pairs:
        try:
            td = nx.shortest_path_length(G, s, g)
        except nx.NetworkXNoPath:
            continue
        hv, _ = get_heuristic(model, s, g, G, mapping, eidx, obstacles)
        err = abs(hv - td)
        errors.append(err)
        true_dists.append(td)
        pred_dists.append(hv)
        print(f"From {s} to {g}: pred={hv:.2f}, true={td}, err={err:.2f}")

    avg_error = np.mean(errors) if errors else float('nan')
    rmse = compute_rmse(true_dists, pred_dists)
    print(f"\nAvg. Abs. Error: {avg_error:.2f}")
    print(f"RMSE: {rmse:.2f}")

    return errors, true_dists, pred_dists

import matplotlib.cm as cm
import matplotlib.pyplot as plt

def visualize_gnn(G, pos, hvec, start, goal):
    # Build ordered list of nodes so hvec indices line up
    node_list = list(G.nodes())
    values = [float(hvec[node]) if isinstance(hvec, dict) else float(hvec[i]) 
              for i, node in enumerate(node_list)]
    
    # Normalize for colormap
    vmin, vmax = min(values), max(values)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.viridis

    # Create figure + axis
    fig, ax = plt.subplots(figsize=(8,8))

    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.2, edge_color='black', ax=ax)
    
    # Draw nodes colored by heuristic
    node_colors = [cmap(norm(values[i])) for i in range(len(node_list))]
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=node_list,
        node_color=node_colors,
        node_size=300,
        edgecolors='black',
        linewidths=0.5,
        ax=ax
    )
    
    # Highlight start and goal
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=[start],
        node_color='lime',
        node_size=350,
        edgecolors='black',
        linewidths=1,
        label='start',
        ax=ax
    )
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=[goal],
        node_color='yellow',
        node_size=350,
        edgecolors='black',
        linewidths=1,
        label='goal',
        ax=ax
    )
    
    # Labels
    labels = {n: str(n) for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)
    ax.text(start[0], start[1]+0.3, 'Start', color='green', fontsize=12, ha='center')
    ax.text(goal[0],  goal[1]+0.3,  'Goal',  color='orange', fontsize=12, ha='center')
    
    # Colorbar
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array(values)
    cbar = fig.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label('Predicted Heuristic')

    ax.legend(scatterpoints=1)
    ax.axis('off')
    ax.set_title("Color indicates the distance to the goal node.")
    
    plt.show()