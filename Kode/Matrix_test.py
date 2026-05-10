import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm
import os
np.random.seed(11)

from Sanity_graf import *
from Matrix_graph import *
from Matrix_simulering import *
from Simulering import save_simulation_results

print("start")
A =  sbm_opinion_graph([5000,5000])
print("matrix gen")
B ,_,_= create_matrix_rep(A)

print("Sim")

start_cond, end_state,all_avg_opinions,all_avg_distances = simple_simulation_matrix(A,100)

save_simulation_results_matrix(start_cond, end_state,all_avg_opinions,all_avg_distances,'Ingen_medier_ingen_disinfo_ikke_dynamiske_kanter','png','resultater/simulering_matrix')
print("done")
