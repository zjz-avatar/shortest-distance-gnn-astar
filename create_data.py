import torch
import networkx as nx
import math

class Sample:
    def __init__(self, x, y, edge_index, mask):
        self.x = x
        self.y = y
        self.edge_index = edge_index
        self.mask = mask

def get_edge_index(G, mapping):
    edges = []
    for u, v in G.edges():
        edges.append((mapping[u], mapping[v]))
        edges.append((mapping[v], mapping[u]))
    return torch.tensor(edges, dtype=torch.long).t().contiguous()

def manhattan_distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def generate_landmark_samples(G, mapping, obstacles, landmarks, distance_threshold):
    
    samples = []
    valid_pairs = []
    DIV = 11

    print(f"\nFiltering landmark pairs by Manhattan distance (max {distance_threshold}):")
    filtered = []
    for i in range(len(landmarks)):
        for j in range(i + 1, len(landmarks)):
            d = manhattan_distance(landmarks[i], landmarks[j])
            if d <= distance_threshold:
                filtered.append((landmarks[i], landmarks[j]))
                print(f"  Including {landmarks[i]} ↔ {landmarks[j]} (d={d})")
            else:
                print(f"  Excluding {landmarks[i]} ↔ {landmarks[j]} (d={d} > {distance_threshold})")
    print(f"Total filtered pairs: {len(filtered)}\n")

    print("Training samples created from landmark paths:")
    for (l1, l2) in filtered:
        for (s, g) in [(l1, l2), (l2, l1)]:
            try:
                path = nx.shortest_path(G, source=s, target=g)
                dists = nx.single_source_dijkstra_path_length(G, g)
            except nx.NetworkXNoPath:
                continue
            valid_pairs.append((s, g))
            print(f"  From {s} to {g} (path len={len(path)-1})")

            x = torch.zeros(len(mapping), 6)
            y = torch.zeros(len(mapping))
            mask = torch.zeros(len(mapping), dtype=torch.bool)

            for n, idx in mapping.items():
                dx, dy = n[0] - g[0], n[1] - g[1]
                r = math.hypot(dx, dy)
                theta = math.atan2(dy, dx) / math.pi
                x[idx] = torch.tensor([dx/DIV, dy/DIV, r/DIV, theta,
                                       float(n not in obstacles),
                                       float(n == g)])
                if n in path:
                    y[idx] = dists[n]
                    mask[idx] = True

            edge_index = get_edge_index(G, mapping)
            samples.append(Sample(x, y, edge_index, mask))

    print(f"\nTotal training samples: {len(samples)}")
    return samples, valid_pairs