import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm
import os
from scipy.stats import truncnorm

from Graf import *

def random_relations_uniform_graph(num_of_nodes,size_of_array):
    Graph =  nx.DiGraph()
    fromnodes, tonodes =  np.random.randint(num_of_nodes, size=(2, size_of_array))
    for node in range(num_of_nodes):
        opinion = np.round(np.random.uniform(-1, 1), 2)
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


def ba_opinion_uniform_graph(n=50, m=2):
    # BA graph — preferential attachment gives you hubs
    G_base = nx.barabasi_albert_graph(n, m)
    G = nx.DiGraph()

    for node in range(n):
        G.add_node(node,
            opinion=np.round(np.random.uniform(-1, 1), 2),
            learningrate=np.random.uniform(0.1, 0.5),
            acceptrate=0,
            type='Human'
        )
    for x, y in G_base.edges():
        weight = round(np.clip(np.random.randn() * 0.5, -1, 1), 2)
        G.add_edge(x, y, weight=weight)
        G.add_edge(y, x, weight=weight)

    return G


def ws_opinion_uniform_graph(n=50, k=4, p=0.1, seed = None,factor = 1.0):
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

    opinions = np.round(np.random.uniform(-1, 1, size=n), 2)
    learning_rates = rng.uniform(0.1, 0.5, size=n)

    for node in range(n):
        G.add_node(node,
            opinion=opinions[node],
            learningrate=learning_rates[node],
            acceptrate=0, type='Human')

    edges = list(G_base.edges())
    w_xy = factor*np.round(truncnorm.rvs(c, d, loc=mean_edge, scale=sd_edge, size=len(edges), random_state=rng), 2)
    w_yx = factor*np.round(truncnorm.rvs(c, d, loc=mean_edge, scale=sd_edge, size=len(edges), random_state=rng), 2)

    for (x, y), wxy, wyx in zip(edges, w_xy, w_yx):
        G.add_edge(x, y, weight=wxy)
        G.add_edge(y, x, weight=wyx)
    return G






def hk_opinion_uniform_graph(n=50, m=2, p=0.5):
    G_base = nx.powerlaw_cluster_graph(n, m, p)
    G = nx.DiGraph()
    for node in range(n):
        G.add_node(node,
            opinion=np.round(np.random.uniform(-1, 1), 2),
            learningrate=np.random.uniform(0.1, 0.5),
            acceptrate=0, type='Human')
    for x, y in G_base.edges():
        weight = round(np.clip(np.random.randn() * 0.5, -1, 1), 2)
        G.add_edge(x, y, weight=weight)
        G.add_edge(y, x, weight=weight)
    return G

def sbm_opinion_uniform_graph(sizes=None, p_in=0.3, p_out=0.02):
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
        opinion = np.round(np.clip(community_bias * 0.5 + np.random.uniform(-1, 1) * 0.3, -1, 1), 2)
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


def random_relations_polar_graph(num_of_nodes,size_of_array):
    Graph =  nx.DiGraph()
    fromnodes, tonodes =  np.random.randint(num_of_nodes, size=(2, size_of_array))
    for node in range(num_of_nodes):
        opinion = float(np.random.choice([-1, 1]))
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


def ba_opinion_polar_graph(n=50, m=2):
    # BA graph — preferential attachment gives you hubs
    G_base = nx.barabasi_albert_graph(n, m)
    G = nx.DiGraph()

    for node in range(n):
        G.add_node(node,
            opinion=float(np.random.choice([-1, 1])),
            learningrate=np.random.uniform(0.1, 0.5),
            acceptrate=0,
            type='Human'
        )
    for x, y in G_base.edges():
        weight = round(np.clip(np.random.randn() * 0.5, -1, 1), 2)
        G.add_edge(x, y, weight=weight)
        G.add_edge(y, x, weight=weight)

    return G


def ws_opinion_polar_graph(n=50, k=4, p=0.1, seed = None,factor = 1.0):
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

    opinions = rng.choice([-1, 1],size = n)
    learning_rates = rng.uniform(0.1, 0.5, size=n)

    for node in range(n):
        G.add_node(node,
            opinion=opinions[node],
            learningrate=learning_rates[node],
            acceptrate=0, type='Human')

    edges = list(G_base.edges())
    w_xy = factor*np.round(truncnorm.rvs(c, d, loc=mean_edge, scale=sd_edge, size=len(edges), random_state=rng), 2)
    w_yx = factor*np.round(truncnorm.rvs(c, d, loc=mean_edge, scale=sd_edge, size=len(edges), random_state=rng), 2)

    for (x, y), wxy, wyx in zip(edges, w_xy, w_yx):
        G.add_edge(x, y, weight=wxy)
        G.add_edge(y, x, weight=wyx)
    return G








def hk_opinion_polar_graph(n=50, m=2, p=0.5):
    G_base = nx.powerlaw_cluster_graph(n, m, p)
    G = nx.DiGraph()
    for node in range(n):
        G.add_node(node,
            opinion=float(np.random.choice([-1, 1])),
            learningrate=np.random.uniform(0.1, 0.5),
            acceptrate=0, type='Human')
    for x, y in G_base.edges():
        weight = round(np.clip(np.random.randn() * 0.5, -1, 1), 2)
        G.add_edge(x, y, weight=weight)
        G.add_edge(y, x, weight=weight)
    return G

def sbm_opinion_polar_graph(sizes=None, p_in=0.3, p_out=0.02):
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
        opinion = float(np.random.choice([-1, 1]))
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






def er_opinion_graph(n=50, p=0.1, seed=None, factor=1.0):
    """
    Creates an Erdős-Rényi random graph using nx.erdos_renyi_graph.
    Each possible edge between n nodes is included independently with probability p.

        Parameters:
            n: int (default 50), how many nodes in the graph
            p: float (between 0 and 1, default 0.1) probability that any given edge exists
        Node-attributes:
            opinion: float (between -1 and 1) randomly assigned through a normal distribution. Represents the opinion of the node
            learningrate: float (between 0.1 and 0.5) how accepting/how easily influenced a node is
            acceptrate: float (0 for disabled or else between 0 and 1) used for dynamic edges. How accepting of differing opinion the node is.
                        how far the other node can be from you in the normal distribution before cutoff
            type: string describes what the node is representing: Human, media or disinformation
        Edge_attribute:
            weight: float (between 0 and 1) normally distributed, how much influence does a node have on another node.
        Returns:
            G: The graph
    """
    rng = np.random.default_rng(seed)
    G_base = nx.erdos_renyi_graph(n, p, seed=seed)
    G = nx.DiGraph()

    # Normally distributed opinions between -1 and 1
    mean_op, sd_op, low_op, high_op = 0, 0.5, -1, 1
    a, b = (low_op - mean_op) / sd_op, (high_op - mean_op) / sd_op

    # Normally distributed edge weights between 0 and 1
    mean_edge, sd_edge, low_edge, high_edge = 0.5, 0.25, 0, 1
    c, d = (low_edge - mean_edge) / sd_edge, (high_edge - mean_edge) / sd_edge

    opinions = np.round(truncnorm.rvs(a, b, loc=mean_op, scale=sd_op, size=n, random_state=rng), 2)
    learning_rates = rng.uniform(0.1, 0.5, size=n)

    for node in range(n):
        G.add_node(node,
            opinion=opinions[node],
            learningrate=learning_rates[node],
            acceptrate=0,
            type='Human')

    edges = list(G_base.edges())
    w_xy = factor * np.round(truncnorm.rvs(c, d, loc=mean_edge, scale=sd_edge, size=len(edges), random_state=rng), 2)
    w_yx = factor * np.round(truncnorm.rvs(c, d, loc=mean_edge, scale=sd_edge, size=len(edges), random_state=rng), 2)

    for (x, y), wxy, wyx in zip(edges, w_xy, w_yx):
        G.add_edge(x, y, weight=wxy)
        G.add_edge(y, x, weight=wyx)

    return G


def er_opinion_uniform_graph(n=50, p=0.1, seed=None, factor=1.0):
    """
    Creates an Erdős-Rényi random graph using nx.erdos_renyi_graph.
    Each possible edge between n nodes is included independently with probability p.

        Parameters:
            n: int (default 50), how many nodes in the graph
            p: float (between 0 and 1, default 0.1) probability that any given edge exists
        Node-attributes:
            opinion: float (between -1 and 1) randomly assigned through a normal distribution. Represents the opinion of the node
            learningrate: float (between 0.1 and 0.5) how accepting/how easily influenced a node is
            acceptrate: float (0 for disabled or else between 0 and 1) used for dynamic edges. How accepting of differing opinion the node is.
                        how far the other node can be from you in the normal distribution before cutoff
            type: string describes what the node is representing: Human, media or disinformation
        Edge_attribute:
            weight: float (between 0 and 1) normally distributed, how much influence does a node have on another node.
        Returns:
            G: The graph
    """
    rng = np.random.default_rng(seed)
    G_base = nx.erdos_renyi_graph(n, p, seed=seed)
    G = nx.DiGraph()

    # Normally distributed opinions between -1 and 1
    mean_op, sd_op, low_op, high_op = 0, 0.5, -1, 1
    a, b = (low_op - mean_op) / sd_op, (high_op - mean_op) / sd_op

    # Normally distributed edge weights between 0 and 1
    mean_edge, sd_edge, low_edge, high_edge = 0.5, 0.25, 0, 1
    c, d = (low_edge - mean_edge) / sd_edge, (high_edge - mean_edge) / sd_edge

    opinions = np.round(rng.uniform(-1, 1, size=n), 2)
    learning_rates = rng.uniform(0.1, 0.5, size=n)

    for node in range(n):
        G.add_node(node,
            opinion=opinions[node],
            learningrate=learning_rates[node],
            acceptrate=0,
            type='Human')

    edges = list(G_base.edges())
    w_xy = factor * np.round(truncnorm.rvs(c, d, loc=mean_edge, scale=sd_edge, size=len(edges), random_state=rng), 2)
    w_yx = factor * np.round(truncnorm.rvs(c, d, loc=mean_edge, scale=sd_edge, size=len(edges), random_state=rng), 2)

    for (x, y), wxy, wyx in zip(edges, w_xy, w_yx):
        G.add_edge(x, y, weight=wxy)
        G.add_edge(y, x, weight=wyx)

    return G

def er_opinion_polar_graph(n=50, p=0.1, seed=None, factor=1.0):
    """
    Creates an Erdős-Rényi random graph using nx.erdos_renyi_graph.
    Each possible edge between n nodes is included independently with probability p.

        Parameters:
            n: int (default 50), how many nodes in the graph
            p: float (between 0 and 1, default 0.1) probability that any given edge exists
        Node-attributes:
            opinion: float (between -1 and 1) randomly assigned through a normal distribution. Represents the opinion of the node
            learningrate: float (between 0.1 and 0.5) how accepting/how easily influenced a node is
            acceptrate: float (0 for disabled or else between 0 and 1) used for dynamic edges. How accepting of differing opinion the node is.
                        how far the other node can be from you in the normal distribution before cutoff
            type: string describes what the node is representing: Human, media or disinformation
        Edge_attribute:
            weight: float (between 0 and 1) normally distributed, how much influence does a node have on another node.
        Returns:
            G: The graph
    """
    rng = np.random.default_rng(seed)
    G_base = nx.erdos_renyi_graph(n, p, seed=seed)
    G = nx.DiGraph()

    # Normally distributed opinions between -1 and 1
    mean_op, sd_op, low_op, high_op = 0, 0.5, -1, 1
    a, b = (low_op - mean_op) / sd_op, (high_op - mean_op) / sd_op

    # Normally distributed edge weights between 0 and 1
    mean_edge, sd_edge, low_edge, high_edge = 0.5, 0.25, 0, 1
    c, d = (low_edge - mean_edge) / sd_edge, (high_edge - mean_edge) / sd_edge

    opinions = rng.choice([-1, 1], size=n)
    learning_rates = rng.uniform(0.1, 0.5, size=n)

    for node in range(n):
        G.add_node(node,
            opinion=opinions[node],
            learningrate=learning_rates[node],
            acceptrate=0,
            type='Human')

    edges = list(G_base.edges())
    w_xy = factor * np.round(truncnorm.rvs(c, d, loc=mean_edge, scale=sd_edge, size=len(edges), random_state=rng), 2)
    w_yx = factor * np.round(truncnorm.rvs(c, d, loc=mean_edge, scale=sd_edge, size=len(edges), random_state=rng), 2)

    for (x, y), wxy, wyx in zip(edges, w_xy, w_yx):
        G.add_edge(x, y, weight=wxy)
        G.add_edge(y, x, weight=wyx)

    return G
