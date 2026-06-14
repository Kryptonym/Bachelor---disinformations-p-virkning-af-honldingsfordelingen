import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm
import os

def ws_opinion_graph(n=50, k=4, p=0.1):
    #small world graf
    G_base = nx.newman_watts_strogatz_graph(n, k, p)
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






def create_matrix_rep(Graf):
    nodes =  sorted(Graf.nodes())
    mat_graf= nx.to_numpy_array(Graf,nodelist = nodes)

    opinions     = np.array([Graf.nodes[n]['opinion']     for n in nodes])
    learningrate = np.array([Graf.nodes[n]['learningrate'] for n in nodes])
    return mat_graf, opinions, learningrate
