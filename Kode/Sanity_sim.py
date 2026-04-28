import numpy as np
np.random.seed(11)
print('Loading functions')
from Sanity_graf import ba_opinion_polar_graph, ba_opinion_uniform_graph, hk_opinion_polar_graph, hk_opinion_uniform_graph, random_relations_polar_graph, random_relations_uniform_graph, sbm_opinion_polar_graph, sbm_opinion_uniform_graph, ws_opinion_polar_graph, ws_opinion_uniform_graph
from Simuleringslogik import *
from Simulering import *
from Graf import *


import matplotlib.pyplot as plt
import networkx as nx
import networkx.utils as nx_utils
print('Done loading functions')

steps = 100
print('Gen, af grafer med normalfordelte holdninger')
Graf_random_normal =  random_relations_graph(1000,150)
Graf_ba_normal = ba_opinion_graph(1000,2)
Graf_ws_normal = ws_opinion_graph(1000)
Graf_hk_normal = hk_opinion_graph(1000)
Graf_sbm_normal = sbm_opinion_graph([1000,1000])

#degreedist(Graf_ba_normal,"spring")

print('Gen, af grafer med uniformfordelte holdninger')
Graf_random_uni =  random_relations_uniform_graph(1000,150)
Graf_ba_uni = ba_opinion_uniform_graph(1000,2)
Graf_hk_uni = hk_opinion_uniform_graph(1000)
Graf_ws_uni = ws_opinion_uniform_graph(1000)
Graf_sbm_uni = sbm_opinion_uniform_graph([1000,1000])

print('Gen, af grafer med polart holdninger')
Graf_random_polar =  random_relations_polar_graph(1000,150)
Graf_ba_polar = ba_opinion_polar_graph(1000,2)
Graf_ws_polar = ws_opinion_polar_graph(1000)
Graf_hk_polar = hk_opinion_polar_graph(1000)
Graf_sbm_polar = sbm_opinion_polar_graph([1000,1000])


print("tilfældig graf")
start_cond, end_state, all_avg_opinions,all_avg_distances = simple_simulation(Graf_random_normal,steps)
save_simulation_results(start_cond, end_state,all_avg_opinions,all_avg_distances,'Random_normal','png','resultater/sanity/normal')
start_cond, end_state, all_avg_opinions,all_avg_distances = simple_simulation(Graf_random_uni,steps)
save_simulation_results(start_cond, end_state,all_avg_opinions,all_avg_distances,'Random_uniform','png','resultater/sanity/uniform')
start_cond, end_state, all_avg_opinions,all_avg_distances = simple_simulation(Graf_random_polar,steps)
save_simulation_results(start_cond, end_state,all_avg_opinions,all_avg_distances,'Random_polar','png','resultater/sanity/polar')


print("Barabási–Albert graf")
start_cond, end_state, all_avg_opinions,all_avg_distances = simple_simulation(Graf_ba_normal,steps)
save_simulation_results(start_cond, end_state,all_avg_opinions,all_avg_distances,'BA_normal','png','resultater/sanity/normal')
start_cond, end_state, all_avg_opinions,all_avg_distances = simple_simulation(Graf_ba_uni,steps)
save_simulation_results(start_cond, end_state,all_avg_opinions,all_avg_distances,'BA_uniform','png','resultater/sanity/uniform')
start_cond, end_state, all_avg_opinions,all_avg_distances = simple_simulation(Graf_ba_polar,steps)
save_simulation_results(start_cond, end_state,all_avg_opinions,all_avg_distances,'BA_polar','png','resultater/sanity/polar')


print("Watts–Strogatz graf (small world)")
start_cond, end_state, all_avg_opinions,all_avg_distances = simple_simulation(Graf_ws_normal,steps)
save_simulation_results(start_cond, end_state,all_avg_opinions,all_avg_distances,'WS_normal','png','resultater/sanity/normal')
start_cond, end_state, all_avg_opinions,all_avg_distances = simple_simulation(Graf_ws_uni,steps)
save_simulation_results(start_cond, end_state,all_avg_opinions,all_avg_distances,'WS_uniform','png','resultater/sanity/uniform')
start_cond, end_state, all_avg_opinions,all_avg_distances = simple_simulation(Graf_ws_polar,steps)
save_simulation_results(start_cond, end_state,all_avg_opinions,all_avg_distances,'WS_polar','png','resultater/sanity/polar')


print("Holme–Kim graf (powerlaw cluster)")
start_cond, end_state, all_avg_opinions,all_avg_distances = simple_simulation(Graf_hk_normal,steps)
save_simulation_results(start_cond, end_state,all_avg_opinions,all_avg_distances,'HK_normal','png','resultater/sanity/normal')
start_cond, end_state, all_avg_opinions,all_avg_distances = simple_simulation(Graf_hk_uni,steps)
save_simulation_results(start_cond, end_state,all_avg_opinions,all_avg_distances,'HK_uniform','png','resultater/sanity/uniform')
start_cond, end_state, all_avg_opinions,all_avg_distances = simple_simulation(Graf_hk_polar,steps)
save_simulation_results(start_cond, end_state,all_avg_opinions,all_avg_distances,'HK_polar','png','resultater/sanity/polar')


print("Stochastic block model graf")
start_cond, end_state, all_avg_opinions,all_avg_distances = simple_simulation(Graf_sbm_normal,steps)
save_simulation_results(start_cond, end_state,all_avg_opinions,all_avg_distances,'SBM_normal','png','resultater/sanity/normal')
start_cond, end_state, all_avg_opinions,all_avg_distances = simple_simulation(Graf_sbm_uni,steps)
save_simulation_results(start_cond, end_state,all_avg_opinions,all_avg_distances,'SBM_uniform','png','resultater/sanity/uniform')
start_cond, end_state, all_avg_opinions,all_avg_distances = simple_simulation(Graf_sbm_polar,steps)
save_simulation_results(start_cond, end_state,all_avg_opinions,all_avg_distances,'SBM_polar','png','resultater/sanity/polar')
