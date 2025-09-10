import math
import heapq
import networkx as nx

def compute_euclidean_heuristics(goal, G):
    """
    Compute Euclidean distance from each node coordinate to the goal.
    `goal` is an (x,y) tuple.
    Returns dict: coord -> distance.
    """
    heuristics = {}
    gx, gy = goal
    for coord in G.nodes():
        dx = coord[0] - gx
        dy = coord[1] - gy
        heuristics[coord] = math.hypot(dx, dy)
    return heuristics

##############################################################################################################

def astar(G, start, goal, obstacles=None):
    """
    Perform A* search on graph G (whose nodes are (x,y) tuples).
    start/goal must be coord tuples.
    Returns:
      - path:     list of coords from start to goal
      - visited:  set of coords that were expanded
    """
    obstacles = set(obstacles or [])
    start_coord, goal_coord = start, goal

    # Precompute heuristics
    h = compute_euclidean_heuristics(goal_coord, G)

    # Frontier: (f = g+h, g, coord)
    frontier = [(h[start_coord], 0.0, start_coord)]
    came_from = {start_coord: None}
    cost_so_far = {start_coord: 0.0}
    visited = set()

    while frontier:
        f, g, current = heapq.heappop(frontier)
        if current in visited:
            continue
        visited.add(current)

        if current == goal_coord:
            break

        for nbr in G.neighbors(current):
            if nbr in obstacles:
                continue
            new_cost = g + 1.0  # all edges have weight 1
            if nbr not in cost_so_far or new_cost < cost_so_far[nbr]:
                cost_so_far[nbr] = new_cost
                priority = new_cost + h[nbr]
                heapq.heappush(frontier, (priority, new_cost, nbr))
                came_from[nbr] = current

    # Reconstruct path
    path = []
    node = goal_coord
    while node is not None:
        path.append(node)
        node = came_from.get(node)
    path.reverse()

    return path, visited

##############################################################################################################

import matplotlib.pyplot as plt
import networkx as nx

def visualize_astar(G, pos, path, visited):
    plt.figure(figsize=(8,8))
    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.2, edge_color='black')
    
    # Partition nodes
    path_set    = set(path)
    visited_set = set(visited)
    
    default_nodes  = [n for n in G.nodes() if n not in visited_set]
    explored_nodes = [n for n in visited_set if n not in path_set]
    path_nodes     = path  # keep order, though draw order doesn't matter
    
    # Draw all categories
    nx.draw_networkx_nodes(G, pos,
                           nodelist=default_nodes,
                           node_color='lightgrey',
                           node_size=300,
                           edgecolors='black',
                           linewidths=0.5)
    nx.draw_networkx_nodes(G, pos,
                           nodelist=explored_nodes,
                           node_color='skyblue',
                           node_size=300,
                           edgecolors='black',
                           linewidths=0.5,
                           label='explored nodes')
    nx.draw_networkx_nodes(G, pos,
                           nodelist=path_nodes,
                           node_color='deepskyblue',
                           node_size=300,
                           edgecolors='black',
                           linewidths=0.5,
                           label='optimal nodes')
    
    # Highlight start and goal
    start, goal = path[0], path[-1]
    nx.draw_networkx_nodes(G, pos,
                           nodelist=[start],
                           node_color='lime',
                           node_size=350,
                           edgecolors='black',
                           linewidths=1,
                           label='start')
    nx.draw_networkx_nodes(G, pos,
                           nodelist=[goal],
                           node_color='yellow',
                           node_size=350,
                           edgecolors='black',
                           linewidths=1,
                           label='goal')
    
    # Label each node with its coordinate
    labels = {n: str(n) for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=8)
    
    # Annotate start/goal text
    plt.text(start[0], start[1]+0.3, 'Start', color='green', fontsize=12, ha='center')
    plt.text(goal[0],  goal[1]+0.3,  'Goal',  color='orange', fontsize=12, ha='center')
    
    plt.legend(scatterpoints=1)
    plt.axis('off')
    plt.title("Optimal Route Using Traditional A*")
    plt.show()