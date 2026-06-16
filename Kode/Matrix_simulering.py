from Matrix_simuleringslogik import *
from Matrix_graph import ws_opinion_graph, create_matrix_rep, GraphState

Test = ws_opinion_graph(10,2,0.1)
Test_matrix = create_matrix_rep(Test)

a = simpel_simulering(Test_matrix,10)

print(a)