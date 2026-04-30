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


def drop_probability(opinion_diff, tolerance):
    # No chance of dropping if within tolerance
    if opinion_diff <= tolerance:
        return 0.0
    # Scales from 0 (just outside tolerance) to 1 (max disagreement of 2.0)
    excess = opinion_diff - tolerance
    max_excess = 2.0 - tolerance  # opinions are presumably in [0,1] or [-1,1]
    return min(excess / max_excess, 1.0)

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
           for pred in preds:
               pred_opinion = G.nodes[pred]['opinion']
               diff = abs(current_node_opinion - pred_opinion)
               if diff > tolerance:
                   prob = drop_probability(diff, tolerance)
                   if random.random() < prob:
                       G.remove_edge(pred, node)
    return G

def update_edges_for_media(Graph, amount_of_std):
    G = Graph.copy()
    all_opinions = [G.nodes[n]['opinion'] for n in G.nodes()]
    opinion_std = np.std(all_opinions)
    tolerance = amount_of_std * opinion_std

    human_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'Human']
    media_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'Media']

    for media in media_nodes:
        media_opinion = G.nodes[media]['opinion']
        current_targets = set(G.successors(media))

        for human in human_nodes:
            if human in current_targets:
                continue
            diff = abs(media_opinion - G.nodes[human]['opinion'])
            if diff < tolerance:
                connect_prob = 1 - (diff / tolerance)  # closer = more likely to connect
                if random.random() < connect_prob:
                    weight = round(1 - diff, 2)
                    G.add_edge(media, human, weight=weight)

    return G


def update_edges_for_disinformation(Graph, amount_of_std):
    G = Graph.copy()
    all_opinions = [G.nodes[n]['opinion'] for n in G.nodes()]
    opinion_std = np.std(all_opinions)
    tolerance = amount_of_std * opinion_std

    human_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'Human']
    disinfo_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'Disinfo']

    for disinfo in disinfo_nodes:
        disinfo_opinion = G.nodes[disinfo]['opinion']
        current_targets = set(G.successors(disinfo))

        for human in human_nodes:
            if human in current_targets:
                continue
            diff = abs(disinfo_opinion - G.nodes[human]['opinion'])
            if diff < tolerance:
                connect_prob = 1 - (diff / tolerance)
                if random.random() < connect_prob:
                    weight = round(1 - diff, 2)
                    G.add_edge(disinfo, human, weight=weight)

    return G
