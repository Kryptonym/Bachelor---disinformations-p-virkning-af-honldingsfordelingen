import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import scipy as sp
from scipy.stats import truncnorm
from tqdm import tqdm
import os
from dataclasses import dataclass



def ws_opinion_graph(n=50, k=4, p=0.1, seed = None):
    """
    Creates a Watts-Strogatz-graph also known as a small world graph / network
    Uses nx.newman_watts_strogatz_graph, so that the connections out of the node is not at the expense of a neighbour. example: a node has k=4 and an extra
    edge out.

        Parameters:
            n: int (default 50), how many nodes in the graph
            k: int  (default 4), how many neighbours a nodes has
            p: float (between 0 and 1, default 0.1) the likelihood that a node has a connection out
        Node-attributes:
            opinion: float (between -1 and 1) randomly assigned through a normal distribution between -1 and 1. Represents the opinion of the node
            learningrate: float (between 0.1 and 0.5) how accepting/ how easy influenced a node is
            acceptrate: float (0 for disabled or else between 0 and 1) used for dynamic edges. How accepting of differing opnion the nodes is.
                        how far the other node can be from you in the normal distribution before cutoff
            type: string describes what the nodes is representing: Human, media or disinformation
        Edge_attribute:
            Weight: Float (between 0 and 1) normally distributed how much influence does a node have on another node.
        Returns:
            G: The graf

    """
    rng = np.random.default_rng(seed)

    G_base = nx.newman_watts_strogatz_graph(n, k, p, seed=seed)
    G = nx.DiGraph()

    # Normalfordelte holdninger mellem -1 og 1
    mean_op, sd_op, low_op, high_op = 0, 0.5, -1, 1
    a, b = (low_op - mean_op) / sd_op, (high_op - mean_op) / sd_op

    # Normalfordelte kantværdier mellem 0 og 1
    mean_edge, sd_edge, low_edge, high_edge = 0.5, 0.25, 0, 1
    c, d = (low_edge - mean_edge) / sd_edge, (high_edge - mean_edge) / sd_edge

    opinions = np.round(truncnorm.rvs(a, b, loc=mean_op, scale=sd_op, size=n, random_state=rng), 2)
    learning_rates = rng.uniform(0.1, 0.5, size=n)

    for node in range(n):
        G.add_node(node,
            opinion=opinions[node],
            learningrate=learning_rates[node],
            acceptrate=0, type='Human')

    edges = list(G_base.edges())
    w_xy = np.round(truncnorm.rvs(c, d, loc=mean_edge, scale=sd_edge, size=len(edges), random_state=rng), 2)
    w_yx = np.round(truncnorm.rvs(c, d, loc=mean_edge, scale=sd_edge, size=len(edges), random_state=rng), 2)

    for (x, y), wxy, wyx in zip(edges, w_xy, w_yx):
        G.add_edge(x, y, weight=wxy)
        G.add_edge(y, x, weight=wyx)
    return G


def add_media_nodes(state, num_of_media_nodes, reach=None, seed=None):
    """
    Adds media nodes to an existing graph.
    Media nodes broadcast to a random (or specified) percent of the population.

    Parameters:
        Graph: existing nx.DiGraph()
        num_of_media_nodes: how many media nodes to add
        reach: float (0-1) to fix reach for all media nodes,
               or None to randomise each media node's reach independently
    """
    Graph = state_to_graph(state)
    rng = np.random.default_rng(seed)


    human_nodes = [n for n, d in Graph.nodes(data=True) if d['type'] == 'Human']
    num_of_humans = len(human_nodes)

    # Normalfordelte holdninger mellem 0 og 1
    mean_op, sd_op, low_op, high_op = 0.5, 0.25, 0, 1
    a, b = (low_op - mean_op) / sd_op, (high_op - mean_op) / sd_op

    # Normalfordelte kantværdier mellem 0 og 1
    mean_edge, sd_edge, low_edge, high_edge = 0.5, 0.25, 0, 1
    c, d = (low_edge - mean_edge) / sd_edge, (high_edge - mean_edge) / sd_edge

    next_id = max(Graph.nodes()) + 1

    for i in range(num_of_media_nodes):
        media_id = next_id + i
        opinion = np.round(truncnorm.rvs(a, b, loc=mean_op, scale=sd_op, size=1, random_state=rng)[0], 2)
        media_reach = reach if reach is not None else rng.uniform(0, 1)
        num_reached = int(np.floor(media_reach * num_of_humans))
        acceptrate = 1 - rng.uniform(0.01, 0.1)

        Graph.add_node(media_id, opinion=opinion, learningrate=0.0, acceptrate=acceptrate,
                        type='Media', reach=np.round(media_reach, 2))

        targets = rng.choice(human_nodes, size=num_reached, replace=False)
        for human in targets:
            weight = np.round(truncnorm.rvs(c, d, loc=mean_edge, scale=sd_edge, size=1, random_state=rng)[0], 2)
            Graph.add_edge(media_id, human, weight=weight)

    return create_matrix_rep(Graph)

def add_disinfo_nodes(state, num_of_disinfo_nodes, reach=None, seed=None):
    """
    Adds disinfo nodes to an existing graph.
    Disinfo nodes broadcast to a random (or specified) percent of the population.

    Parameters:
        Graph: existing nx.DiGraph()
        num_of_disinfo_nodes: how many disinfonodes nodes to add
        reach: float (0-1) to fix reach for all media nodes,
               or None to randomise each media node's reach independently
    """

    Graph = state_to_graph(state)
    rng = np.random.default_rng(seed)

    # Normalfordelte holdninger mellem -1 og 1
    mean_op, sd_op, low_op, high_op = -0.5, 0.25, -1, 0
    a, b = (low_op - mean_op) / sd_op, (high_op - mean_op) / sd_op

    # Normalfordelte kantværdier mellem 0 og 1
    mean_edge, sd_edge, low_edge, high_edge = 0.5, 0.25, 0, 1
    c, d = (low_edge - mean_edge) / sd_edge, (high_edge - mean_edge) / sd_edge


    human_nodes = [n for n, d in Graph.nodes(data=True) if d['type'] == 'Human']
    media_nodes = [n for n, d in Graph.nodes(data=True) if d['type'] == 'Media']
    num_of_humans = len(human_nodes)
    num_of_media = len(media_nodes)
    num_of_total = num_of_humans + num_of_media

    next_id = max(Graph.nodes()) + 1


    for i in range(num_of_disinfo_nodes):
        disinfo_id = next_id + i
        opinion = np.round(truncnorm.rvs(a, b, loc=mean_op, scale=sd_op, size=1, random_state=rng)[0], 2)

        media_reach = reach if reach is not None else rng.uniform(0, 0.05)
        num_reached = int(np.floor(media_reach * num_of_humans))
        acceptrate  = 1 - rng.uniform(0.01, 0.1)

        Graph.add_node(disinfo_id, opinion=opinion, learningrate=0.0,acceptrate= acceptrate,
                       type='Disinformation', reach=round(media_reach, 2))

        targets = rng.choice(num_of_humans, size=num_reached, replace=False)
        for human in targets:
            weight = np.round(truncnorm.rvs(c, d, loc=mean_edge, scale=sd_edge, size=1, random_state=rng)[0], 2)
            Graph.add_edge(disinfo_id, human, weight=weight)

    return create_matrix_rep(Graph)



@dataclass
class GraphState:
    matrix: np.ndarray
    opinions: np.ndarray
    learningrate: np.ndarray
    acceptrate: np.ndarray
    type: np.ndarray

def create_matrix_rep(Graf):
    nodes = sorted(Graf.nodes())
    matrix_graf_rep     = nx.to_numpy_array(Graf, nodelist=nodes, weight="weight")
    opinions_vector     = np.array([Graf.nodes[n]['opinion']      for n in nodes])
    learningrate_vector = np.array([Graf.nodes[n]['learningrate'] for n in nodes])
    acceptrate_vector   = np.array([Graf.nodes[n]['acceptrate']   for n in nodes])
    type_vector         = np.array([Graf.nodes[n]['type']         for n in nodes])

    return GraphState(
        matrix=matrix_graf_rep,
        opinions=opinions_vector,
        learningrate=learningrate_vector,
        acceptrate=acceptrate_vector,
        type=type_vector,
    )


def state_to_graph(state):
    n = len(state.opinions)
    G = nx.from_numpy_array(state.matrix, create_using=nx.DiGraph)
    for i in range(n):
        G.nodes[i]['opinion']      = state.opinions[i]
        G.nodes[i]['learningrate'] = state.learningrate[i]
        G.nodes[i]['acceptrate']   = state.acceptrate[i]
        G.nodes[i]['type']         = state.type[i]
    return G





def add_media_nodes_full_normal_dist(state, num_of_media_nodes, reach=None, seed=None):
    """
    Adds media nodes to an existing graph.
    Media nodes broadcast to a random (or specified) percent of the population.

    Parameters:
        Graph: existing nx.DiGraph()
        num_of_media_nodes: how many media nodes to add
        reach: float (0-1) to fix reach for all media nodes,
               or None to randomise each media node's reach independently
    """
    Graph = state_to_graph(state)
    rng = np.random.default_rng(seed)


    human_nodes = [n for n, d in Graph.nodes(data=True) if d['type'] == 'Human']
    num_of_humans = len(human_nodes)

    # Normalfordelte holdninger mellem 0 og 1
    mean_op, sd_op, low_op, high_op = 0, 0.5, -1, 1
    a, b = (low_op - mean_op) / sd_op, (high_op - mean_op) / sd_op

    # Normalfordelte kantværdier mellem 0 og 1
    mean_edge, sd_edge, low_edge, high_edge = 0.5, 0.25, 0, 1
    c, d = (low_edge - mean_edge) / sd_edge, (high_edge - mean_edge) / sd_edge

    next_id = max(Graph.nodes()) + 1

    for i in range(num_of_media_nodes):
        media_id = next_id + i
        opinion = np.round(truncnorm.rvs(a, b, loc=mean_op, scale=sd_op, size=1, random_state=rng)[0], 2)
        media_reach = reach if reach is not None else rng.uniform(0, 1)
        num_reached = int(np.floor(media_reach * num_of_humans))
        acceptrate = 1 - rng.uniform(0.01, 0.1)

        Graph.add_node(media_id, opinion=opinion, learningrate=0.0, acceptrate=acceptrate,
                        type='Media', reach=np.round(media_reach, 2))

        targets = rng.choice(human_nodes, size=num_reached, replace=False)
        for human in targets:
            weight = np.round(truncnorm.rvs(c, d, loc=mean_edge, scale=sd_edge, size=1, random_state=rng)[0], 2)
            Graph.add_edge(media_id, human, weight=weight)

    return create_matrix_rep(Graph)