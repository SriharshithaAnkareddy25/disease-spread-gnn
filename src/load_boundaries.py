"""Load Indian district boundaries and compute centroids."""
import geopandas as gpd

BOUNDARY_PATH = "data/raw/boundaries/datameet-maps/Districts/Census_2011/2011_Dist.shp"


def load_districts() -> gpd.GeoDataFrame:
    gdf = gpd.read_file(BOUNDARY_PATH)
    gdf["centroid"] = gdf.geometry.centroid
    return gdf


if __name__ == "__main__":
    districts = load_districts()
    print(districts.head())
    print(f"{len(districts)} districts loaded")
