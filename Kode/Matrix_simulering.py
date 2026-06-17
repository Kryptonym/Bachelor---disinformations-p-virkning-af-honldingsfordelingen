from Matrix_graph import ws_opinion_graph, create_matrix_rep, GraphState
from Matrix_simuleringslogik import time_step_no_media_no_disinfo, simpel_simulering, data_processing
import numpy as np
import matplotlib.pyplot as plt
print('start')
TestGraf =  ws_opinion_graph(1000,10,0.1,seed = 11)

MatrixGraf = create_matrix_rep(TestGraf)

MatrixGraf.opinions, no_media_no_disinfo_sim_results,no_media_no_disinfo_startcond,no_media_no_disinfo_endcond= simpel_simulering(MatrixGraf,1000)

average_opinionresult, average_distance_to_the_mean_result = data_processing(no_media_no_disinfo_sim_results)

print('plot')

plt.figure()
plt.plot(np.arange(len(average_opinionresult)),average_opinionresult,'b-',label = 'Average opinion')
plt.plot(np.arange(len(average_opinionresult)),np.zeros(len(average_opinionresult)),'r--',label = 'Zero')
plt.legend()
plt.grid()
plt.show()

plt.figure()
plt.plot(np.arange(len(average_distance_to_the_mean_result)),average_distance_to_the_mean_result,'b-',label = 'Average opinion')
plt.plot(np.arange(len(average_distance_to_the_mean_result)),np.zeros(len(average_opinionresult)),'r--',label = 'Zero')
plt.legend()
plt.grid()
plt.title('Average distance to the average opinion')
plt.show()


plt.figure()
plt.hist(no_media_no_disinfo_startcond,bins =100)
plt.title('Starting opinions')
plt.xlim(-1, 1)
plt.show()

plt.figure()
plt.hist(no_media_no_disinfo_endcond,bins =100)
plt.title('Ending opinions')
plt.xlim()
plt.show()


plt.figure()
plt.hist(no_media_no_disinfo_startcond, bins=100, alpha=0.5, label='Start')
plt.hist(no_media_no_disinfo_endcond, bins=100, alpha=0.5, label='End')
plt.xlim(-1, 1)
plt.legend()
plt.title('Opinion distribution: start vs end')
plt.show()