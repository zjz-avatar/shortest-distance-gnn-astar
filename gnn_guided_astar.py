import torch
import math
import networkx as nx
import matplotlib.pyplot as plt

def compute_heuristics_for_goal(goal, model, G, mapping, edge_index, obstacles, division):
    """
    Computes heuristic predictions from all nodes to the goal using the GNN.
    Returns a dict: {node: heuristic value}
    """
    model.eval()
    with torch.no_grad():
        x = torch.zeros(len(mapping), 6)
        for n, idx in mapping.items():
            dx, dy = n[0] - goal[0], n[1] - goal[1]
            r = math.hypot(dx, dy)
            theta = math.atan2(dy, dx) / math.pi
            is_obs = float(n not in obstacles)
            is_goal = float(n == goal)
            x[idx] = torch.tensor([dx / division, dy / division, r / division, theta, is_obs, is_goal])

        pred = model(x, edge_index).squeeze()
        heuristics = {n: pred[mapping[n]].item() for n in mapping}
        return heuristics


def a_star_with_gnn(G, start, goal, model, mapping, edge_index, obstacles, division):
    import heapq

    # Precompute heuristics from all nodes to the goal
    heuristics = compute_heuristics_for_goal(goal, model, G, mapping, edge_index, obstacles, division)

    frontier = [(0, start)]
    came_from = {start: None}
    cost_so_far = {start: 0}
    visited = set()

    while frontier:
        _, current = heapq.heappop(frontier)
        visited.add(current)

        if current == goal:
            break

        for neighbor in G.neighbors(current):
            new_cost = cost_so_far[current] + 1
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                h = heuristics.get(neighbor, float('inf'))
                priority = new_cost + h
                heapq.heappush(frontier, (priority, neighbor))
                came_from[neighbor] = current

    # Reconstruct path
    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = came_from.get(node)
    path.reverse()

    return path, visited

def visualize_astar_exploration(G, path, visited, start, goal, obstacles):
    pos = {n: n for n in G.nodes()}
    fig, ax = plt.subplots(figsize=(16, 16))
    
    # Draw full graph
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.2)
    nx.draw_networkx_nodes(G, pos, node_color='lightgrey', node_size=200, ax=ax)

    # Explored nodes (visited)
    nx.draw_networkx_nodes(G, pos, nodelist=list(visited), node_color='skyblue', node_size=250, ax=ax)

    # Final path
    nx.draw_networkx_nodes(G, pos, nodelist=path, node_color='deepskyblue', node_size=300, ax=ax)

    # Start and goal
    nx.draw_networkx_nodes(G, pos, nodelist=[start], node_color='lime', node_size=400, edgecolors='black', ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=[goal], node_color='yellow', node_size=400, edgecolors='black', ax=ax)
    plt.text(start[0], start[1]+0.3, 'Start', color='green', fontsize=12, ha='center')
    plt.text(goal[0], goal[1]+0.3, 'Goal', color='orange', fontsize=12, ha='center')

    # Obstacles
    visible_obstacles = [n for n in obstacles if n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, nodelist=visible_obstacles, node_color='black', node_size=250, ax=ax)

    ax.set_title("GNN-Guided A* Search: Explored Nodes and Path")
    ax.axis('off')
    plt.show()