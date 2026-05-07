import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm
import os
from Sanity_graf import ws_opinion_graph
from Matrix_graph import create_matrix_rep


A =  ws_opinion_graph()

B ,_,_= create_matrix_rep(A)

print(B)
