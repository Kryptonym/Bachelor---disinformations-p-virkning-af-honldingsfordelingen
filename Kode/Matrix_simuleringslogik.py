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

    for i in range(timesteps):
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








