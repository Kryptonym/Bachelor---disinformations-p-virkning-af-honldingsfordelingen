import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm
from Simuleringslogik import *
import os
import copy


def simple_simulation(Graf, Num_of_steps):
    start_cond =  copy.deepcopy(Graf)
    all_avg_opinions =  {}
    all_avg_distances = {}
    G = copy.deepcopy(Graf)
    for step in tqdm(range(Num_of_steps), desc="Simulating"):
        G, avg_opinion = update_opinion(G)
        all_avg_opinions[step] = avg_opinion
        opinions = list(nx.get_node_attributes(G, 'opinion').values())
        avg_distance = np.mean([abs(op - avg_opinion) for op in opinions])
        all_avg_distances[step] = avg_distance
    end_state = copy.deepcopy(G)

    return start_cond, end_state, all_avg_opinions,all_avg_distances


def simulation_with_dynamic_human_edges(Graf, Num_of_steps,amount_of_std):
    start_cond =  copy.deepcopy(Graf)
    all_avg_opinions =  {}
    all_avg_distances = {}
    G = copy.deepcopy(Graf)
    human_nodes = [n for n, d in G.nodes(data=True) if d['type'] == 'Human']
    for step in tqdm(range(Num_of_steps), desc="Simulating"):
        G, avg_opinion = update_opinion(G)
        G = update_edges_for_person(G, amount_of_std)
        all_avg_opinions[step] = avg_opinion
        opinions = [G.nodes[n]['opinion'] for n in human_nodes]
        avg_distance = np.mean([abs(op - avg_opinion) for op in opinions])
        all_avg_distances[step] = avg_distance
        G = update_edges_for_person(G, amount_of_std)
    end_state =copy.deepcopy(G)

    return start_cond, end_state, all_avg_opinions,all_avg_distances



def show_simulation_results(start_cond,end_state,all_avg_opinions,all_avg_distances):
    start_opinions = list(nx.get_node_attributes(start_cond, 'opinion').values())
    plt.hist(start_opinions, bins=200, range=(-1, 1))
    plt.xlabel('Opinion')
    plt.ylabel('Count')
    plt.title('Distribution of opinions of the start condition')
    plt.show()


    end_opinions = list(nx.get_node_attributes(end_state, 'opinion').values())
    plt.hist(end_opinions, bins=200, range=(-1, 1))
    plt.xlabel('Opinion')
    plt.ylabel('Count')
    plt.title('Distribution of opinions at the end of the simulation')
    plt.show()


    plt.figure()
    plt.plot(all_avg_opinions.keys(), all_avg_opinions.values(),color = 'k',label = 'Avg. opinion')
    plt.xlabel('Step')
    plt.ylabel('Average Opinion')
    plt.title('Average opinioin over time')
    plt.axhline(0, color='red', linestyle='--', linewidth=1,label = 'neutral reference')  # neutral reference line
    plt.grid()
    plt.legend()
    plt.show()


    plt.figure()
    plt.plot(all_avg_distances.keys(), all_avg_distances.values(),color = 'k',label = 'Avg. distance')
    plt.xlabel('Step')
    plt.ylabel('Average distance')
    plt.title('Average distance to the average (Polarisation)')
    plt.grid()
    plt.legend()
    plt.show()


def save_simulation_results(start_cond,end_state,all_avg_opinions,all_avg_distances,filename="simulation", fmt="png",folder="resultater"):
    """
        Parameters:
            filename : str - base name for saved files (without extension), e.g. "run_01"
            fmt      : str - file format, e.g. "png", "svg", "pdf"
    """
    os.makedirs(folder, exist_ok=True)
    plt.figure()
    start_opinions = list(nx.get_node_attributes(start_cond, 'opinion').values())
    plt.hist(start_opinions, bins=200, range=(-1, 1))
    plt.xlabel('Opinion')
    plt.ylabel('Count')
    plt.title('Distribution of opinions of the start condition')
    plt.savefig(os.path.join(folder, f"{filename}_start_distribution.{fmt}"), format=fmt, bbox_inches='tight')
    plt.close()


    plt.figure()
    end_opinions = list(nx.get_node_attributes(end_state, 'opinion').values())
    plt.hist(end_opinions, bins=200, range=(-1, 1))
    plt.xlabel('Opinion')
    plt.ylabel('Count')
    plt.title('Distribution of opinions at the end of the simulation')
    plt.savefig(os.path.join(folder, f"{filename}_end_distribution.{fmt}"), format=fmt, bbox_inches='tight')
    plt.close()


    plt.figure()
    plt.plot(all_avg_opinions.keys(), all_avg_opinions.values(),color = 'k',label = 'Avg. opinion')
    plt.xlabel('Step')
    plt.ylabel('Average Opinion')
    plt.title('Average opinioin over time')
    plt.axhline(0, color='red', linestyle='--', linewidth=1,label = 'neutral reference')  # neutral reference line
    plt.grid()
    plt.legend()
    plt.savefig(os.path.join(folder, f"{filename}_avg_opinion.{fmt}"), format=fmt, bbox_inches='tight')
    plt.close()


    plt.figure()
    plt.plot(all_avg_distances.keys(), all_avg_distances.values(),color = 'k',label = 'Avg. distance')
    plt.xlabel('Step')
    plt.ylabel('Average distance')
    plt.title('Average distance to the average (Polarisation)')
    plt.grid()
    plt.legend()
    plt.savefig(os.path.join(folder, f"{filename}_avg_abs_distance_polarisation.{fmt}"), format=fmt, bbox_inches='tight')
    plt.close()
