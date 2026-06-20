from Matrix_graph import ws_opinion_graph, create_matrix_rep, GraphState, add_media_nodes,add_disinfo_nodes
from Matrix_simuleringslogik import time_step_no_media_no_disinfo, simpel_simulering, data_processing
import numpy as np
import matplotlib.pyplot as plt
TestGraf =  ws_opinion_graph(1000,15,0.1,seed = 11)

MatrixGraf = create_matrix_rep(TestGraf)

MatrixGraf.opinions, no_media_no_disinfo_sim_results,no_media_no_disinfo_startcond,no_media_no_disinfo_endcond= simpel_simulering(MatrixGraf,15)

mask = MatrixGraf.type == 'Human'

average_opinionresult, average_distance_to_the_mean_result = data_processing(no_media_no_disinfo_sim_results,mask)


plt.figure()
plt.hist(no_media_no_disinfo_startcond, bins=100, alpha=0.5, label='Start')
plt.hist(no_media_no_disinfo_endcond, bins=100, alpha=0.5, label='End')
plt.xlim(-1, 1)
plt.legend()
plt.title('Opinion distribution: start vs end')
plt.show()


MatrixGrafMedia=add_media_nodes(MatrixGraf,10,seed =11)
maskmedia = MatrixGrafMedia.type == 'Human'
MatrixGrafMedia.opinions, media_no_disinfo_sim_results,media_no_disinfo_startcond,media_no_disinfo_endcond= simpel_simulering(MatrixGrafMedia,30)
media_average_opinionresult, media_average_distance_to_the_mean_result = data_processing(media_no_disinfo_sim_results,maskmedia)



plt.figure()
plt.hist(media_no_disinfo_startcond, bins=100, alpha=0.5, label='Start')
plt.hist(media_no_disinfo_endcond, bins=100, alpha=0.5, label='End')
plt.xlim(-1, 1)
plt.legend()
plt.title('Opinion distribution: start vs end')
plt.show()

MatrixGrafdisinfo=add_disinfo_nodes(MatrixGraf,10,seed =11)
maskdisinfo = MatrixGrafdisinfo.type == 'Human'
MatrixGrafdisinfo.opinions, media_disinfo_sim_results,media_disinfo_startcond,media_disinfo_endcond= simpel_simulering(MatrixGrafdisinfo,40)
media_disinfo_average_opinionresult, media_disinfo_average_distance_to_the_mean_result = data_processing(media_disinfo_sim_results,maskdisinfo)




plt.figure()
plt.hist(media_disinfo_startcond, bins=100, alpha=0.5, label='Start')
plt.hist(media_disinfo_endcond, bins=100, alpha=0.5, label='End')
plt.xlim(-1, 1)
plt.legend()
plt.title('Opinion distribution: start vs end')
plt.show()






totalgennemsnit = np.concatenate((average_opinionresult,media_average_opinionresult,media_disinfo_average_opinionresult))

totalgennemgennemsnit = np.concatenate((average_distance_to_the_mean_result,media_average_distance_to_the_mean_result,media_disinfo_average_distance_to_the_mean_result))



plt.figure()
plt.plot(np.arange(len(totalgennemsnit)),totalgennemsnit,'b-',label = 'Average opinion')
plt.plot(np.arange(len(totalgennemsnit)),np.zeros(len(totalgennemsnit)),'r--',label = 'Zero')
plt.legend()
plt.grid()
plt.show()

plt.figure()
plt.plot(np.arange(len(totalgennemgennemsnit)),totalgennemgennemsnit,'b-',label = 'Average opinion')
plt.plot(np.arange(len(totalgennemgennemsnit)),np.zeros(len(totalgennemgennemsnit)),'r--',label = 'Zero')
plt.legend()
plt.grid()
plt.title('Average distance to the average opinion')
plt.show()