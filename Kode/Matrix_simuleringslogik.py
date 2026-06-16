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
    matrix = Graph_state.matrix.copy()
    opinions =Graph_state.opinions.copy()
    learningrate = Graph_state.learningrate.copy()
    acceptrate =Graph_state.acceptrate.copy()
    type_vector =Graph_state.type

    weight_sum = matrix.sum(axis=0)
    ops_through_time =  []
    
    for i in range(timesteps):
        opinions =time_step_no_media_no_disinfo(matrix,opinions,learningrate,weight_sum)
        ops_through_time.append(opinions)
    return ops_through_time