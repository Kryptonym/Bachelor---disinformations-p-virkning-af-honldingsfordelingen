import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# --- Build a sample graph ---
G = nx.karate_club_graph()

# Detect communities via Louvain-style greedy modularity
from networkx.algorithms.community import greedy_modularity_communities
communities = list(greedy_modularity_communities(G))
community_map = {}
for i, comm in enumerate(communities):
    for node in comm:
        community_map[node] = i

# Node attributes
degree = dict(G.degree())
max_deg = max(degree.values())

# Layout
pos = nx.spring_layout(G, seed=42, k=0.6)

# Colors per community
palette = ["#4361EE", "#F72585", "#4CC9F0", "#7209B7", "#3A0CA3"]
node_colors = [palette[community_map[n] % len(palette)] for n in G.nodes()]
node_sizes  = [200 + 900 * (degree[n] / max_deg) for n in G.nodes()]

# --- Figure ---
fig, ax = plt.subplots(figsize=(11, 8))
fig.patch.set_facecolor("#0D1117")
ax.set_facecolor("#0D1117")

# Edges
nx.draw_networkx_edges(
    G, pos, ax=ax,
    edge_color="#FFFFFF", alpha=0.12, width=0.9
)

# Nodes
nx.draw_networkx_nodes(
    G, pos, ax=ax,
    node_color=node_colors,
    node_size=node_sizes,
    alpha=0.95,
    linewidths=0.6,
    edgecolors="#FFFFFF"
)

# Labels for high-degree nodes only
hub_labels = {n: str(n) for n in G.nodes() if degree[n] >= 8}
nx.draw_networkx_labels(
    G, pos, labels=hub_labels, ax=ax,
    font_size=7, font_color="white", font_weight="bold"
)

# Legend
legend_handles = [
    mpatches.Patch(color=palette[i % len(palette)], label=f"Community {i+1}")
    for i in range(len(communities))
]
legend = ax.legend(
    handles=legend_handles,
    loc="upper left",
    framealpha=0.25,
    facecolor="#1C1F26",
    edgecolor="#444",
    labelcolor="white",
    fontsize=8,
    title="Communities",
    title_fontsize=8,
)
legend.get_title().set_color("white")

# Stats annotation
n_nodes = G.number_of_nodes()
n_edges = G.number_of_edges()
ax.text(
    0.99, 0.01,
    f"Nodes: {n_nodes}   Edges: {n_edges}   Communities: {len(communities)}",
    transform=ax.transAxes,
    ha="right", va="bottom",
    fontsize=7.5, color="#AAAAAA"
)

ax.set_title("Karate Club Network — Community Detection",
             color="white", fontsize=14, fontweight="bold", pad=14)
ax.axis("off")
plt.tight_layout()
plt.show()
# plt.savefig("/mnt/user-data/outputs/networkx_graph.png", dpi=160, bbox_inches="tight",  facecolor=fig.get_facecolor())
print("Saved.")
