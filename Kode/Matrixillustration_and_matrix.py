from Sanity_graf import ws_opinion_graph
from Graf import degreedist
from Matrix_graph import create_matrix_rep

eksempelgraf =  ws_opinion_graph(10,3,0.1)
degreedist(eksempelgraf,"circular")

matrix, opinions, learningrate  = create_matrix_rep(eksempelgraf)
print(matrix)
print(opinions)
print(learningrate)
