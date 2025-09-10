import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from create_data import get_edge_index
from torch_geometric.nn import SAGEConv

# ===== GraphSAGE GNN Model =====
class GraphSAGENet(nn.Module):
    def __init__(self, in_dim=6, hid=64, n_layers=6):
        super().__init__()
        self.convs = nn.ModuleList()
        # input layer
        self.convs.append(SAGEConv(in_dim, hid))
        # hidden layers
        for _ in range(n_layers - 2):
            self.convs.append(SAGEConv(hid, hid))
        # output layer
        self.convs.append(SAGEConv(hid, 1))

    def forward(self, x, edge_index):
        # first layer
        x = self.convs[0](x, edge_index)
        x = F.relu(x)
        # hidden layers with skip
        for conv in self.convs[1:-1]:
            out = conv(x, edge_index)
            x = F.relu(out) + x
        # output
        return self.convs[-1](x, edge_index).squeeze(-1)

# ===== Training =====
def train_model(data_samples, lr=1e-3, epochs=101, lambda_consistency=1.0):
    random.seed(42)
    torch.manual_seed(42)

    model = GraphSAGENet()
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
            print(f"Epoch {epoch}, Loss: {total_loss/len(data_samples):.4f}")

    return model
