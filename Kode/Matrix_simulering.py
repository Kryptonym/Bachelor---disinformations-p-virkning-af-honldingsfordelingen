from Matrix_simuleringslogik import *
from Matrix_graph import *
def simple_simulation_matrix(Graf, Num_of_steps):
    A, opinions, lr = create_matrix_rep(Graf)

    start_opinions = opinions.copy()
    all_avg_opinions  = {}
    all_avg_distances = {}

    for step in tqdm(range(Num_of_steps), desc="Simulating"):
        opinions, avg_opinion = update_opinion_matrix(A, opinions, lr)
        all_avg_opinions[step]  = avg_opinion
        all_avg_distances[step] = np.mean(np.abs(opinions - avg_opinion))

    return start_opinions, opinions, all_avg_opinions, all_avg_distances



def save_simulation_results_matrix(start_cond,end_state,all_avg_opinions,all_avg_distances,filename="simulation", fmt="png",folder="resultater"):
    """
        Parameters:
            filename : str - base name for saved files (without extension), e.g. "run_01"
            fmt      : str - file format, e.g. "png", "svg", "pdf"
    """
    os.makedirs(folder, exist_ok=True)
    plt.figure()
    plt.hist(start_cond, bins=200, range=(-1, 1))
    plt.xlabel('Opinion')
    plt.ylabel('Count')
    plt.title('Distribution of opinions of the start condition')
    plt.savefig(os.path.join(folder, f"{filename}_start_distribution.{fmt}"), format=fmt, bbox_inches='tight')
    plt.close()


    plt.figure()
    plt.hist(end_state, bins=200, range=(-1, 1))
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
