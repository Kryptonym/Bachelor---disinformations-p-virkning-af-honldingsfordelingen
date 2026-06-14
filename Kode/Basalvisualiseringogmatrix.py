import networkx as nx
import matplotlib.pyplot as plt
from Graf import illustration_of_graph, ws_opinion_graph
from Matrix_graph import create_matrix_rep

matrix = ws_opinion_graph(10,4,0.1)

a,b,c = create_matrix_rep(matrix)

nx.draw(matrix, with_labels=True)
plt.show()

illustration_of_graph(matrix)

print(a)
