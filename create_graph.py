import networkx as nx
import matplotlib.pyplot as plt

def build_grid_graph():
    width, height = 11, 11
    G = nx.grid_2d_graph(width, height)
    obstacles = [(9, y) for y in range(1, 10)] \
                + [(x, 1) for x in range(1, 9)] \
                + [(x, 9) for x in range(1, 9)]
    G.remove_nodes_from(obstacles)
    mapping = {n: i for i, n in enumerate(G.nodes())}
    inv_map = {i: n for n, i in mapping.items()}
    return G, mapping, inv_map, obstacles

##############################################################################################################
def visualize_graph(G, mapping, inv_map, start, goal):

    # Positions are just the (x,y) tuples
    pos = {n: n for n in G.nodes()}

    plt.figure(figsize=(8,8))
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    nx.draw_networkx_nodes(
        G, pos,
        node_color='lightgrey',
        node_size=300,
        linewidths=0.5,
        edgecolors='black'
    )

    # Label each node with its coordinate
    labels = {n: str(n) for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=8)

    nx.draw_networkx_nodes(
        G, pos,
        nodelist=[start],
        node_color='lime',
        node_size=400,
        label='start'
    )

    nx.draw_networkx_nodes(
        G, pos,
        nodelist=[goal],
        node_color='yellow',
        node_size=400,
        label='goal'
    )

    plt.legend(scatterpoints=1)
    plt.axis('off')
    plt.title("Grid Graph with Obstacles, Choose your Start (lime) and Goal (orange) nodes")
    plt.show()
