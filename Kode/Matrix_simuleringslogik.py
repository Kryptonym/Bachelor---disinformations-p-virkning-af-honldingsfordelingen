import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm
import os
from Matrix_graph import ws_opinion_graph,create_matrix_rep, GraphState



def time_step_no_media_no_disinfo(matrix,opinions,learningrate,weight_sum):
    """
    Calculates the new opinion for timestep t+1.
    new_opinions = opinions + learningrate * (avg_neighbour_opinion - opinions)
    Parameters
    """
    weighted_neighbour_opinion_sum = matrix.T @ opinions
    avg_neighbour_opinion = np.divide(weighted_neighbour_opinion_sum,
                                      weight_sum, out=np.zeros_like(weighted_neighbour_opinion_sum),
                                      where=weight_sum != 0)
    new_opinions = opinions + learningrate * (avg_neighbour_opinion - opinions)

    return new_opinions

def simpel_simulering(Graph_state,timesteps):
    """
    Calculates the new opinion for timestep t+1.
    new_opinions = opinions + learningrate * (avg_neighbour_opinion - opinions)
    Parameters
    """
     
    matrix = Graph_state.matrix.copy()
    opinions =Graph_state.opinions.copy()
    learningrate = Graph_state.learningrate.copy()
    acceptrate =Graph_state.acceptrate.copy()
    type_vector =Graph_state.type

    startcondition = Graph_state.opinions.copy()




    weight_sum = matrix.sum(axis=0)
    ops_through_time =  []

    for i in tqdm(range(timesteps), desc="Simulating"):
        opinions =time_step_no_media_no_disinfo(matrix,opinions,learningrate,weight_sum)
        ops_through_time.append(opinions)


    endcondition = ops_through_time[-1]

    return opinions, ops_through_time, startcondition, endcondition


def data_processing(ops_through_time,mask):
    """
    Calculates the new opinion for timestep t+1.
    new_opinions = opinions + learningrate * (avg_neighbour_opinion - opinions)
    Parameters
    """

    ops_through_time = np.array(ops_through_time)

    filtered_ops = ops_through_time[:, mask]


    average_opinion =  filtered_ops.mean(axis = 1)

    average_distance_to_the_mean = np.abs(ops_through_time - average_opinion[:, None]).mean(axis=1)

    return average_opinion, average_distance_to_the_mean





def in_and_out_degree_sum(Graph_state, node_type=None):
    """
    Returns the in-degree and out-degree weight sums for each node.
    Parameters:
        Graph_state: GraphState
        node_type: str or None. If None, returns all nodes.
                   If 'Human', 'Media', or 'Disinformation', filters to that type.
    Returns:
        indegree:  np.ndarray, sum of incoming edge weights per node
        outdegree: np.ndarray, sum of outgoing edge weights per node
    """
    matrix = Graph_state.matrix

    if node_type is not None:
        mask = Graph_state.type == node_type
        submatrix = matrix[np.ix_(mask, mask)]
    else:
        submatrix = matrix

    indegree  = submatrix.sum(axis=0)  # sum over rows → incoming per column (node)
    outdegree = submatrix.sum(axis=1)  # sum over cols → outgoing per row (node)

    return indegree, outdegree



def cross_type_degree(Graph_state, source_type, target_type):
    """
    Returns the summed edge weights from source_type nodes to target_type nodes.
    For each target node: how much total influence it receives from source_type.
    For each source node: how much total influence it sends to target_type.

    Parameters:
        Graph_state: GraphState
        source_type: str, e.g. 'Media' or 'Disinformation'
        target_type: str, e.g. 'Human'
    Returns:
        influence_received: np.ndarray, shape (num_targets,) — total incoming weight per target node
        influence_sent:     np.ndarray, shape (num_sources,) — total outgoing weight per source node
    """
    source_mask = Graph_state.type == source_type
    target_mask = Graph_state.type == target_type

    # submatrix[i, j] = weight from source_i to target_j
    submatrix = Graph_state.matrix[np.ix_(source_mask, target_mask)]

    influence_received = submatrix.sum(axis=0)  # per target: sum over all sources
    influence_sent     = submatrix.sum(axis=1)  # per source: sum over all targets

    return influence_received, influence_sent

