import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm
import os

def create_matrix_rep(Graf):
    nodes =  sorted(Graf.nodes())
    mat_graf= nx.to_numpy_array(Graf,nodelist = nodes)

    opinions     = np.array([Graf.nodes[n]['opinion']     for n in nodes])
    learningrate = np.array([Graf.nodes[n]['learningrate'] for n in nodes])
    return mat_graf, opinions, learningrate
