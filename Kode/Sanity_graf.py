import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm
import os

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

def ws_opinion_uniform_graph(n=50, k=4, p=0.1):
    #small world graf
    G_base = nx.watts_strogatz_graph(n, k, p)
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

def ws_opinion_polar_graph(n=50, k=4, p=0.1):
    #small world graf
    G_base = nx.watts_strogatz_graph(n, k, p)
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
