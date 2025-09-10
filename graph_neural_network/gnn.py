import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from create_data import get_edge_index

# ===== Traditional GNN Model =====
class TraditionalGNN(nn.Module):
    def __init__(self, in_dim=6, hid=64, n_layers=6):
        super().__init__()
        self.n_layers = n_layers
        self.linears = nn.ModuleList()
        self.linears.append(nn.Linear(in_dim, hid))   # Input layer
        for _ in range(n_layers - 2):
            self.linears.append(nn.Linear(hid, hid))  # Hidden layers
        self.linears.append(nn.Linear(hid, 1))        # Output layer

    def forward(self, x, edge_index):
        row, col = edge_index

        # First layer (no skip connection)
        agg = torch.zeros_like(x)
        agg.index_add_(0, row, x[col])
        deg = torch.bincount(row, minlength=x.size(0)).clamp(min=1).unsqueeze(1)
        x = agg / deg
        x = F.relu(self.linears[0](x))

        # Hidden layers with skip connections
        for layer in self.linears[1:-1]:
            agg = torch.zeros_like(x)
            agg.index_add_(0, row, x[col])
            deg = torch.bincount(row, minlength=x.size(0)).clamp(min=1).unsqueeze(1)
            x_agg = agg / deg
            x = F.relu(layer(x_agg)) + x  # skip connection

        # Final layer
        agg = torch.zeros_like(x)
        agg.index_add_(0, row, x[col])
        deg = torch.bincount(row, minlength=x.size(0)).clamp(min=1).unsqueeze(1)
        x = agg / deg
        return self.linears[-1](x).squeeze(-1)

# ===== Training =====
def train_model(data_samples, lr=1e-3, epochs=101, lambda_consistency=1.0):
    """
    Train the TraditionalGNN on provided samples. Returns the trained model.
    """
    # Set seeds for reproducibility
    random.seed(42)
    torch.manual_seed(42)

    model = TraditionalGNN()
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