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



steps = 200
Graf =  random_relations_graph(1000,500)
degreedist(Graf,"circular")
print('Simulering: Ingen medier ingen disinfo ikke dynamiske kanter')
start_cond, end_state, all_avg_opinions,all_avg_distances = simple_simulation(Graf,steps)
#show_simulation_results(start_cond, end_state, all_avg_opinions,all_avg_distances)
save_simulation_results(start_cond, end_state,all_avg_opinions,all_avg_distances,'Ingen_medier_ingen_disinfo_ikke_dynamiske_kanter','png')
A,B,C,D =start_cond, end_state, all_avg_opinions,all_avg_distances

print('Simulering: Ingen medier ingen disinfo dynamiske kanter')
start_cond, end_state, all_avg_opinions,all_avg_distances = simulation_with_dynamic_human_edges(Graf,steps,1)
#show_simulation_results(start_cond, end_state, all_avg_opinions,all_avg_distances)
save_simulation_results(start_cond, end_state,all_avg_opinions,all_avg_distances,'Ingen_medier_ingen_disinfo_dynamiske_kanter','png')
print(nx_utils.graphs_equal(A, start_cond))
print(nx_utils.graphs_equal(B,end_state ))
print(C == all_avg_opinions)
print(D == all_avg_distances)



Graf = add_media_nodes(Graf,10,1)
print('Simulering:  medier ingen disinfo ikke dynamiske kanter')
start_cond, end_state, all_avg_opinions,all_avg_distances = simple_simulation(Graf,steps)
save_simulation_results(start_cond, end_state,all_avg_opinions,all_avg_distances,'medier_ingen_disinfo_ikke_dynamiske_kanter','png')

print('Simulering:  medier ingen disinfo dynamiske kanter')
start_cond, end_state, all_avg_opinions,all_avg_distances = simulation_with_dynamic_human_edges(Graf,steps,1)
save_simulation_results(start_cond, end_state,all_avg_opinions,all_avg_distances,'medier_ingen_disinfo_dynamiske_kanter','png')


Graf = add_disinfo_nodes(Graf,2)


print('done')
