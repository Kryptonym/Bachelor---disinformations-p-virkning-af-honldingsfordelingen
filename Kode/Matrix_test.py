import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm
import os
np.random.seed(11)

from Sanity_graf import *
from Matrix_graph import *
from Matrix_simulering import *
from Simulering import *

print("start")
A =  ws_opinion_graph(10,5,0.1)
print("matrix gen")
B ,C,D= create_matrix_rep(A)

degreedist(A,"circular")
degreedist(B,"circular")

print("Sim")
start_cond, end_state,all_avg_opinions,all_avg_distances = simple_simulation_matrix(A,100)



save_simulation_results_matrix(start_cond, end_state,all_avg_opinions,all_avg_distances,'Ingen_medier_ingen_disinfo_ikke_dynamiske_kanter','png','resultater/simulering_matrix')
print("done")
print("Print B_0")
print(B[0])
print("Print B")
print(B)
print("Print C")
print(C)
print("Print D")
print(D)
print("Print start cond")
print(start_cond)
print("Print end cond")
print(end_state)
