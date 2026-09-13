"""Build the district node table and graph variants (geographic adjacency, gravity model).

Node source: Datameet 2011 Census district boundaries (shapefile).
Attribute source: nishusharma1608/India-Census-2011-Analysis district census CSV.
Join key: shapefile `censuscode` == census CSV `District code` (640/641 districts match;
district name strings are NOT used as the join key since they mismatch across sources).
"""
import geopandas as gpd
import pandas as pd
import networkx as nx
import numpy as np
import scipy.sparse as sp

BOUNDARY_PATH = "data/raw/boundaries/datameet-maps/Districts/Census_2011/2011_Dist.shp"
CENSUS_PATH = "data/raw/census/india-districts-census-2011.csv"

# Albers Equal Area Conic centered on India — used for area (km^2) and centroid
# distance calculations, since the source shapefile is in unprojected EPSG:4326
# (degrees), where area/distance math would be wrong.
INDIA_AEA = "+proj=aea +lat_1=8 +lat_2=32 +lat_0=0 +lon_0=80 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs"


def build_nodes() -> gpd.GeoDataFrame:
    gdf = gpd.read_file(BOUNDARY_PATH)
    census = pd.read_csv(CENSUS_PATH)

    gdf_proj = gdf.to_crs(INDIA_AEA)
    gdf["area_km2"] = gdf_proj.geometry.area / 1e6
    gdf["centroid_proj"] = gdf_proj.geometry.centroid

    nodes = gdf.merge(
        census[["District code", "Population"]],
        left_on="censuscode",
        right_on="District code",
        how="inner",
    )
    nodes["density_per_km2"] = nodes["Population"] / nodes["area_km2"]
    return nodes


def build_geographic_adjacency(nodes: gpd.GeoDataFrame) -> nx.Graph:
    """Edge between two districts if their polygons share a border.

    Uses `.intersects()` rather than `.touches()`: adjacent polygons in this
    shapefile don't share perfectly coincident edges — digitization leaves
    microscopic (~1e-15 area) overlap slivers at shared borders, which makes
    `.touches()` report `False` (it requires exactly zero intersection area)
    even for textbook-adjacent districts (verified e.g. Allahabad/Chitrakoot).
    """
    g = nx.Graph()
    g.add_nodes_from(nodes["censuscode"])

    sindex = nodes.sindex
    for i, row in nodes.iterrows():
        candidate_idx = list(sindex.intersection(row.geometry.bounds))
        for j in candidate_idx:
            if j <= i:
                continue
            other = nodes.iloc[j]
            if row.geometry.intersects(other.geometry):
                g.add_edge(row["censuscode"], other["censuscode"])
    return g


def build_gravity_edges(nodes: gpd.GeoDataFrame) -> pd.DataFrame:
    """Dense gravity-model edge weights: w_ij = (pop_i * pop_j) / dist_ij^2 (meters)."""
    records = []
    codes = nodes["censuscode"].tolist()
    pops = nodes["Population"].tolist()
    centroids = nodes["centroid_proj"].tolist()

    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            dist = centroids[i].distance(centroids[j])
            if dist == 0:
                continue
            weight = (pops[i] * pops[j]) / (dist ** 2)
            records.append((codes[i], codes[j], weight))

    return pd.DataFrame(records, columns=["src", "dst", "weight"])


def sparsify_gravity_edges(gravity_df: pd.DataFrame, k: int = 6) -> pd.DataFrame:
    """Keep each node's top-k heaviest gravity edges, then union both directions.

    A global weight threshold/percentile would strand remote, low-population
    districts entirely (their weights are systematically small vs. dense
    metro pairs, regardless of who their "real" neighbors are), so per-node
    top-k is used instead — every node keeps edges to its k strongest partners.
    Union (not intersection) of both endpoints' top-k, since gravity is
    symmetric but "is in my top-k" isn't: a small district is very likely in
    a huge neighbor's top-k far less often than the reverse.
    """
    directed = pd.concat(
        [
            gravity_df.rename(columns={"src": "a", "dst": "b"}),
            gravity_df.rename(columns={"src": "b", "dst": "a"}),
        ],
        ignore_index=True,
    )
    top_k = directed.sort_values("weight", ascending=False).groupby("a").head(k)
    pair = pd.DataFrame(
        {
            "src": top_k[["a", "b"]].min(axis=1),
            "dst": top_k[["a", "b"]].max(axis=1),
            "weight": top_k["weight"],
        }
    )
    return pair.drop_duplicates(subset=["src", "dst"]).reset_index(drop=True)


def save_adjacency_matrix(edges_df: pd.DataFrame, node_order: list, path: str, weighted: bool) -> None:
    """Save a symmetric sparse adjacency matrix, row/col i == node_order[i] (censuscode)."""
    index = {code: i for i, code in enumerate(node_order)}
    rows = edges_df["src"].map(index).to_numpy()
    cols = edges_df["dst"].map(index).to_numpy()
    values = edges_df["weight"].to_numpy() if weighted else np.ones(len(edges_df))

    n = len(node_order)
    mat = sp.coo_matrix((values, (rows, cols)), shape=(n, n))
    mat = mat + mat.T  # symmetric: edges_df stores each undirected edge once
    sp.save_npz(path, mat.tocsr())


if __name__ == "__main__":
    gdf_raw = gpd.read_file(BOUNDARY_PATH)
    nodes = build_nodes()
    print(f"{len(nodes)} nodes after join (dropped {len(gdf_raw) - len(nodes)} unmatched)")

    nodes.drop(columns=["centroid_proj"]).to_file("data/processed/nodes.geojson", driver="GeoJSON")
    nodes[["censuscode", "DISTRICT", "ST_NM", "Population", "area_km2", "density_per_km2"]].to_csv(
        "data/processed/nodes.csv", index=False
    )
    # Row/col order that data/processed/*_adjacency.npz matrices are indexed by.
    node_order = nodes["censuscode"].tolist()
    pd.Series(node_order, name="censuscode").to_csv("data/processed/node_order.csv", index=True, index_label="row")

    adj_graph = build_geographic_adjacency(nodes)
    adj_edges = pd.DataFrame(adj_graph.edges(), columns=["src", "dst"])
    adj_edges["weight"] = 1
    adj_edges.to_csv("data/processed/edges_geographic_adjacency.csv", index=False)
    save_adjacency_matrix(adj_edges, node_order, "data/processed/geographic_adjacency.npz", weighted=False)
    print(f"Geographic adjacency: {adj_graph.number_of_edges()} edges")

    gravity_df = build_gravity_edges(nodes)
    gravity_df.to_csv("data/processed/edges_gravity_dense.csv", index=False)
    print(f"Gravity model (dense): {len(gravity_df)} pairwise edges")

    gravity_sparse = sparsify_gravity_edges(gravity_df, k=6)
    gravity_sparse.to_csv("data/processed/edges_gravity.csv", index=False)
    save_adjacency_matrix(gravity_sparse, node_order, "data/processed/gravity_adjacency.npz", weighted=True)
    print(f"Gravity model (top-6 sparsified): {len(gravity_sparse)} edges")
