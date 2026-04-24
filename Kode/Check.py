import numpy as np
np.random.seed(11)
print('Loading functions')
from Simuleringslogik import *
from Simulering import *
from Graf import *

import matplotlib.pyplot as plt
import networkx as nx
import networkx.utils as nx_utils
print('Done loading functions')

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



Graf_random =  random_relations_graph(50,50)

Graf_ba = ba_opinion_graph()

Graf_ws = ws_opinion_graph()

Graf_kh = hk_opinion_graph()

Graf_sbm = sbm_opinion_graph()

show_opinion_histogram(Graf_random)
degreedist(Graf_random,"spring")

show_opinion_histogram(Graf_ba)
degreedist(Graf_ba,"spring")

show_opinion_histogram(Graf_ws)
degreedist(Graf_ws,"spring")

show_opinion_histogram(Graf_kh)
degreedist(Graf_kh,"spring")

show_opinion_histogram(Graf_sbm)
degreedist(Graf_sbm,"spring")
