import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm
import os

def update_opinion_matrix(A, opinions, learningrates):
    # Row-normalize by absolute weight sum (same as total_weight in your loop)
    abs_row_sums = np.abs(A).sum(axis=1, keepdims=True)

    # Avoid division by zero (nodes with no predecessors keep their opinion)
    safe_sums = np.where(abs_row_sums == 0, 1, abs_row_sums)
    A_norm = A / safe_sums

    # Weighted average of predecessor opinions
    new_opinions = A_norm @ opinions

    # Blend: old + lr * (new - old)
    blended = opinions + learningrates * (new_opinions - opinions)

    # Clip and round
    updated = np.round(np.clip(blended, -1, 1), 2)

    # Nodes with no predecessors keep old opinion (same as your loop)
    no_pred = (abs_row_sums.squeeze() == 0)
    updated[no_pred] = opinions[no_pred]

    avg_opinion = updated.mean()
    return updated, avg_opinion
