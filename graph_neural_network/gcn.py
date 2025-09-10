import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from create_data import get_edge_index
from torch_geometric.nn import GCNConv
from graph_neural_network.utils import evaluate_model, compute_rmse

# ===== GCN-based GNN Model =====
class GCNNet(nn.Module):
    def __init__(self, in_dim=6, hid=64, n_layers=6):
        super().__init__()
        self.convs = nn.ModuleList()
        # input layer
        self.convs.append(GCNConv(in_dim, hid))
        # hidden layers
        for _ in range(n_layers - 2):
            self.convs.append(GCNConv(hid, hid))
        # output layer
        self.convs.append(GCNConv(hid, 1))

    def forward(self, x, edge_index):
        # first layer
        x = self.convs[0](x, edge_index)
        x = F.relu(x)
        # hidden layers with skip connections
        for conv in self.convs[1:-1]:
            out = conv(x, edge_index)
            x = F.relu(out) + x
        # final layer
        x = self.convs[-1](x, edge_index).squeeze(-1)
        return x

# ===== Inference =====
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

# ===== Training =====
def train_model(data_samples, lr=1e-3, epochs=101, lambda_consistency=1.0):
    random.seed(42)
    torch.manual_seed(42)

    model = GCNNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for sample in data_samples:
            x_noisy = sample.x.clone()
            x_noisy[:, :4] += 0.01 * torch.randn_like(x_noisy[:, :4])
            pred = model(x_noisy, sample.edge_index)

            loss_mse = F.mse_loss(pred[sample.mask], sample.y[sample.mask])
            loss_reg = 0.001 * (pred**2).mean()
            u, v = sample.edge_index
            violation = F.relu(pred[u] - (pred[v] + 1.0))
            loss_consist = violation.mean()

            loss = loss_mse + loss_reg + lambda_consistency * loss_consist
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 10 == 0:
            avg_loss = total_loss / len(data_samples)
            print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")

    return model
