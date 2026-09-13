"""Plot district boundaries with geographic-adjacency edges overlaid."""
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

nodes = gpd.read_file("data/processed/nodes.geojson")
edges = pd.read_csv("data/processed/edges_geographic_adjacency.csv")

centroids = nodes.set_index("censuscode").geometry.centroid

fig, ax = plt.subplots(figsize=(10, 12))
nodes.plot(ax=ax, color="whitesmoke", edgecolor="lightgray", linewidth=0.3)

for _, row in edges.iterrows():
    if row.src not in centroids.index or row.dst not in centroids.index:
        continue
    p1, p2 = centroids[row.src], centroids[row.dst]
    ax.plot([p1.x, p2.x], [p1.y, p2.y], color="steelblue", linewidth=0.3, alpha=0.6)

centroids.plot(ax=ax, color="firebrick", markersize=2)
ax.set_title("India district geographic-adjacency graph (nodes = district centroids)")
ax.set_axis_off()
plt.tight_layout()
plt.savefig("data/processed/geographic_adjacency_graph.png", dpi=200)
print("Saved data/processed/geographic_adjacency_graph.png")
