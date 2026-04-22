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
        learningrate = np.random.uniform(0.01, 0.1)
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
