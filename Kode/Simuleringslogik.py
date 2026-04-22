import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm
import os

def update_opinion(Graph,learning_rate=0.1):
    Graph = Graph.copy()
    new_opinions = {}
    for node in Graph.nodes():
            predecessors = list(Graph.predecessors(node))

            if not predecessors:
                new_opinions[node] = Graph.nodes[node]['opinion']
                continue
            weighted_opinions = []
            weights = []
            for pred in predecessors:
                edge_weight = Graph[pred][node]['weight']
                pred_opinion = Graph.nodes[pred]['opinion']
                weighted_opinions.append(edge_weight * pred_opinion)
                weights.append(abs(edge_weight))
            total_weight = sum(weights)
            if total_weight == 0:
                new_opinions[node] = Graph.nodes[node]['opinion']
            else:
                old_opinion = Graph.nodes[node]['opinion']
                new_opinion = sum(weighted_opinions) / total_weight
                blended = old_opinion + Graph.nodes[node]['learningrate'] * (new_opinion - old_opinion)
                new_opinions[node] = round(np.clip(blended, -1, 1), 2)

    for node, opinion in new_opinions.items():
        Graph.nodes[node]['opinion'] = opinion
    Average_opinion = sum(new_opinions.values())/len(new_opinions)

    return Graph , Average_opinion


def update_edges_for_person(Graph,amount_of_std):
    G = Graph.copy()
    all_opinions = [G.nodes[n]['opinion'] for n in G.nodes()]
    opinion_std = np.std(all_opinions)
    tolerance = amount_of_std * opinion_std

    for node in G.nodes():
           if G.nodes[node].get('type') != 'Human':
               continue
           current_node_opinion = G.nodes[node]['opinion']

           # --- Find new connections ---
           current_preds = set(G.predecessors(node))
           candidates = {}
           for pred in current_preds:
               if G.nodes[pred].get('type') != 'Human':
                   continue
               for pred_of_pred in G.predecessors(pred):
                   if pred_of_pred == node:
                       continue
                   if pred_of_pred in current_preds:
                       continue
                   if G.has_edge(pred_of_pred, node):
                       continue
                   if G.nodes[pred_of_pred].get('type') != 'Human':
                       continue
                   candidates[pred_of_pred] = G.nodes[pred_of_pred]['opinion']  # ← inside inner loop

           if candidates:
               best = min(candidates, key=lambda n: abs(current_node_opinion - candidates[n]))
               best_opinion = candidates[best]
               if abs(current_node_opinion - best_opinion) < tolerance:
                   initial_weight = round(1 - abs(current_node_opinion - best_opinion), 2)
                   G.add_edge(best, node, weight=initial_weight)

           # --- Drop far away connections ---
           preds = list(G.predecessors(node))
           for pred in preds:  # ← same indent level as preds = list(...)
               pred_opinion = G.nodes[pred]['opinion']
               if abs(current_node_opinion - pred_opinion) > tolerance:
                   G.remove_edge(pred, node)
    return G

def update_edges_for_media(Graph,amount_of_std):
    G = Graph.copy()
    nodes =  Graph.nodes()

    return None


def update_edges_for_disinformation(Graph,amount_of_std):
    G = Graph.copy()
    nodes =  Graph.nodes()

    return None
