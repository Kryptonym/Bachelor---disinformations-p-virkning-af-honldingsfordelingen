import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm
import os


def random_relations_graph(num_of_nodes,size_of_array):
    Graph =  nx.DiGraph()
    fromnodes, tonodes =  np.random.randint(num_of_nodes, size=(2, size_of_array))
    for node in range(num_of_nodes):
        opinion = np.round(np.clip(np.random.randn() * 0.5, -1, 1), 2)
        learningrate = np.random.uniform(0.1, 1)
        Graph.add_node(node, opinion=opinion, learningrate=learningrate, acceptrate = 0, type ='Human')


    for x,y in zip(fromnodes,tonodes):
        if x==y:
            pass
        else:
            relationship_type = np.abs(np.random.randn())
            if  relationship_type > 1.96:
                Graph.add_edge(x,y, weight = np.round(np.clip(np.random.randn() * 0.5, -1, 1), 2))
                Graph.add_edge(y,x, weight = np.round(np.clip(np.random.randn() * 0.5, -1, 1), 2))

            else:
                weight = round(np.clip(np.random.randn() * 0.5, -1, 1), 2)
                Graph.add_edge(x,y, weight = weight)
                Graph.add_edge(y,x, weight = weight)
    return Graph


def ba_opinion_graph(n=50, m=2):
    # BA graph — preferential attachment gives you hubs
    G_base = nx.barabasi_albert_graph(n, m)
    G = nx.DiGraph()

    for node in range(n):
        G.add_node(node,
            opinion=np.round(np.clip(np.random.randn() * 0.5, -1, 1), 2),
            learningrate=np.random.uniform(0.1, 0.5),
            acceptrate=0,
            type='Human'
        )
    for x, y in G_base.edges():
        weight = round(np.clip(np.random.randn() * 0.5, -1, 1), 2)
        G.add_edge(x, y, weight=weight)
        G.add_edge(y, x, weight=weight)

    return G

def ws_opinion_graph(n=50, k=4, p=0.1):
    #small world graf
    G_base = nx.watts_strogatz_graph(n, k, p)
    G = nx.DiGraph()
    for node in range(n):
        G.add_node(node,
            opinion=np.round(np.clip(np.random.randn() * 0.5, -1, 1), 2),
            learningrate=np.random.uniform(0.1, 0.5),
            acceptrate=0, type='Human')
    for x, y in G_base.edges():
        weight = round(np.clip(np.random.randn() * 0.5, -1, 1), 2)
        G.add_edge(x, y, weight=weight)
        G.add_edge(y, x, weight=weight)
    return G



def hk_opinion_graph(n=50, m=2, p=0.5):
    G_base = nx.powerlaw_cluster_graph(n, m, p)
    G = nx.DiGraph()
    for node in range(n):
        G.add_node(node,
            opinion=np.round(np.clip(np.random.randn() * 0.5, -1, 1), 2),
            learningrate=np.random.uniform(0.1, 0.5),
            acceptrate=0, type='Human')
    for x, y in G_base.edges():
        weight = round(np.clip(np.random.randn() * 0.5, -1, 1), 2)
        G.add_edge(x, y, weight=weight)
        G.add_edge(y, x, weight=weight)
    return G

def sbm_opinion_graph(sizes=None, p_in=0.3, p_out=0.02):
    if sizes is None:
        sizes = [25, 25]          # two equal communities
    n_groups = len(sizes)
    p = [[p_in  if i == j else p_out
          for j in range(n_groups)]
         for i in range(n_groups)]
    G_base = nx.stochastic_block_model(sizes, p)
    G = nx.DiGraph()
    for node in range(sum(sizes)):
        block = G_base.nodes[node]['block']
        # Seed opinions per community for polarisation
        community_bias = (block / (n_groups - 1) * 2 - 1) if n_groups > 1 else 0
        opinion = np.round(np.clip(
            community_bias * 0.5 + np.random.randn() * 0.3, -1, 1), 2)
        G.add_node(node,
            opinion=opinion,
            learningrate=np.random.uniform(0.1, 0.5),
            acceptrate=0, type='Human',
            community=block)
    for x, y in G_base.edges():
        weight = round(np.clip(np.random.randn() * 0.5, -1, 1), 2)
        G.add_edge(x, y, weight=weight)
        G.add_edge(y, x, weight=weight)
    return G





def add_media_nodes(Graph, num_of_media_nodes, reach=None):
    """
    Adds media nodes to an existing graph.
    Media nodes broadcast to a random (or specified) percent of the population.

    Parameters:
        Graph: existing nx.DiGraph()
        num_of_media_nodes: how many media nodes to add
        reach: float (0-1) to fix reach for all media nodes,
               or None to randomise each media node's reach independently
    """
    human_nodes = [n for n, d in Graph.nodes(data=True) if d['type'] == 'Human']
    num_of_humans = len(human_nodes)


    for i in range(num_of_media_nodes):
        media_id = num_of_humans + i
        opinion = np.round(np.clip(np.random.randn() * 0.5, -0.5, 0.5), 2)

        media_reach = reach if reach is not None else np.random.uniform(0, 1)
        num_reached = int(np.floor(media_reach * num_of_humans))
        acceptrate  = 1 - np.random.uniform(0.01, 0.1)

        Graph.add_node(media_id, opinion=opinion, learningrate=0.0,acceptrate= acceptrate,
                       type='Media', reach=round(media_reach, 2))

        targets = np.random.choice(num_of_humans, size=num_reached, replace=False)
        for human in targets:
            weight = np.round(np.clip(np.random.randn() * 0.5, -1, 1), 2)
            Graph.add_edge(media_id, human, weight=weight)

    return Graph

def add_disinfo_nodes(Graph, num_of_disinfo_nodes, reach=None):
    """
    Adds disinfo nodes to an existing graph.
    Disinfo nodes broadcast to a random (or specified) percent of the population.

    Parameters:
        Graph: existing nx.DiGraph()
        num_of_disinfo_nodes: how many disinfonodes nodes to add
        reach: float (0-1) to fix reach for all media nodes,
               or None to randomise each media node's reach independently
    """
    human_nodes = [n for n, d in Graph.nodes(data=True) if d['type'] == 'Human']
    media_nodes = [n for n, d in Graph.nodes(data=True) if d['type'] == 'Media']
    num_of_humans = len(human_nodes)
    num_of_media = len(media_nodes)
    num_of_total = num_of_humans + num_of_media

    for i in range(num_of_disinfo_nodes):
        disinfo_id = num_of_total + i
        opinion = np.round(np.clip(np.random.randn() * 0.5, -0.5, 0.5), 2)+0.5

        media_reach = reach if reach is not None else np.random.uniform(0, 0.05)
        num_reached = int(np.floor(media_reach * num_of_humans))
        acceptrate  = 1 - np.random.uniform(0.01, 0.1)

        Graph.add_node(disinfo_id, opinion=opinion, learningrate=0.0,acceptrate= acceptrate,
                       type='Media', reach=round(media_reach, 2))

        targets = np.random.choice(num_of_humans, size=num_reached, replace=False)
        for human in targets:
            weight = np.round(np.clip(np.random.randn() * 0.5, -1, 1), 2)
            Graph.add_edge(disinfo_id, human, weight=weight)

    return Graph




def illustration_of_graph(Graph):
    pos = nx.spring_layout(Graph, seed=42)

    nx.draw_networkx_nodes(Graph, pos,
                           node_color="steelblue",
                           node_size=100)

    nx.draw_networkx_labels(Graph, pos,
                            font_color="white",
                            font_size=10)

    nx.draw_networkx_edges(Graph, pos,
                           arrows=True,
                           connectionstyle="arc3,rad=0.2",  # <-- curves the edges apart
                           arrowsize=15)

    edge_labels = nx.get_edge_attributes(Graph, "weight")
    nx.draw_networkx_edge_labels(Graph, pos,
                                 edge_labels=edge_labels,
                                 font_size=8,
                                 font_color="crimson",
                                 label_pos=0.3)  # <-- offset label from center

    plt.show()


def show_opinion_histogram(Graph):
    opinions = list(nx.get_node_attributes(Graph, 'opinion').values())
    plt.hist(opinions, bins=50, range=(-1, 1))
    plt.xlabel('Opinion')
    plt.ylabel('Count')
    plt.title('Distribution of Opinions')
    plt.show()

def show_change(before,after):
    for node in before.nodes():
        print(f"Node {node}: {before.nodes[node]['opinion']} → {after.nodes[node]['opinion']}")



def degreedist(graph,drawstyle):
    """
   The code is taken from the networkx documentation examples, and then modified to suit the needs of the project
   https://networkx.org/documentation/stable/auto_examples/drawing/plot_degree.html#

    Args:
        Graf: a graph made with networkx
        drawstyle (string): can be "spring" or "circular"

    Returns:
        A plot of the graph\\
        A plot of degree rank \\
        A histogram of the degree distribution
    """
    is_directed = graph.is_directed()
    degree_sequence = sorted((d for n, d in graph.degree()), reverse=True)
    dmax = max(degree_sequence)
    fig = plt.figure("Degree of a random graph", figsize=(8, 8))
    axgrid = fig.add_gridspec(5, 4)

    ax0 = fig.add_subplot(axgrid[0:3, :])
    if is_directed:
        components = sorted(nx.weakly_connected_components(graph), key=len, reverse=True)
        title = "Largest Weakly Connected Component"
    else:
        components = sorted(nx.connected_components(graph), key=len, reverse=True)
        title = "Largest Connected Component"

    Gcc = graph.subgraph(graph.subgraph(components[0]))
    if drawstyle == "spring":
        pos = nx.spring_layout(Gcc)
        nx.draw_networkx_nodes(Gcc, pos, ax=ax0, node_size=20)
        nx.draw_networkx_edges(Gcc, pos, ax=ax0, alpha=0.4)
        ax0.set_title("Connected components of Graph")
        ax0.set_axis_off()
    if drawstyle == "circular":
        pos = nx.circular_layout(Gcc)
        nx.draw_networkx_nodes(Gcc, pos, ax=ax0, node_size=20)
        nx.draw_networkx_edges(Gcc, pos, ax=ax0, alpha=0.4)
        ax0.set_title("Connected components of Graph")
        ax0.set_axis_off()

    if is_directed:

        in_degrees = sorted((d for n, d in graph.in_degree()), reverse=True)
        out_degrees = sorted((d for n, d in graph.out_degree()), reverse=True)

        # In-Degree Histogram
        ax1 = fig.add_subplot(axgrid[3:, :2])
        ax1.bar(*np.unique(in_degrees, return_counts=True), color='skyblue', alpha=0.7)
        ax1.set_title("In-Degree Histogram")
        ax1.set_xlabel("Degree")
        ax1.set_ylabel("# of Nodes")

        # Out-Degree Histogram
        ax2 = fig.add_subplot(axgrid[3:, 2:])
        ax2.bar(*np.unique(out_degrees, return_counts=True), color='orange', alpha=0.7)
        ax2.set_title("Out-Degree Histogram")
        ax2.set_xlabel("Degree")

    else:

        degree_sequence = sorted((d for n, d in graph.degree()), reverse=True)

        ax1 = fig.add_subplot(axgrid[3:, :2])
        ax1.plot(degree_sequence, '.')
        ax1.set_title("Degree Rank Plot")

        ax2 = fig.add_subplot(axgrid[3:, 2:])
        ax2.bar(*np.unique(degree_sequence, return_counts=True))
        ax2.set_title("Degree Histogram")


    fig.tight_layout()
    plt.show()
