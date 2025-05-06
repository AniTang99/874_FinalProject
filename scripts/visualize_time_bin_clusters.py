import geopandas as gpd
import pandas as pd
import folium
from folium.features import GeoJsonTooltip
import os

# === CONFIGURATION ===
geojson_path = "data/raw/spd_dispatch_neighborhoods.geojson"  # ⬅️ Replace this with your actual path
cluster_dir = "output/time_bin_clustering"
output_map_path = os.path.join(cluster_dir, "time_bin_clusters_map.html")

# Load neighborhood geometry
gdf = gpd.read_file(geojson_path)
gdf = gdf.rename(columns={"neighborhood": "Neighborhood"})
gdf['Neighborhood'] = gdf['Neighborhood'].str.lower().str.strip()
gdf = gdf.to_crs(epsg=4326)

# Time bins to visualize
bins = ["Late Night", "Early Morning", "Morning", "Afternoon", "Evening", "Night"]
colors = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#e6ab02', '#a6761d', '#666666']

# Create map
m = folium.Map(location=[47.6, -122.33], zoom_start=12, tiles="cartodbpositron")

for i, time_bin in enumerate(bins):
    csv_path = os.path.join(cluster_dir, f"{time_bin.lower().replace(' ', '_')}_clusters.csv")
    if not os.path.exists(csv_path):
        print(f"⚠️ Skipping {time_bin}: File not found.")
        continue

    # Load cluster data
    cluster_df = pd.read_csv(csv_path)
    cluster_df["Neighborhood"] = cluster_df["Dispatch Neighborhood"].str.lower().str.strip()

    # Merge with GeoDataFrame
    merged = gdf.merge(cluster_df, on="Neighborhood", how="left")
    merged["cluster"] = merged["cluster"].fillna(-1).astype(int)

    def style_fn(feature):
        cluster = feature["properties"]["cluster"]
        color = colors[cluster % len(colors)] if cluster != -1 else "#cccccc"
        return {
            "fillColor": color,
            "color": "black",
            "weight": 1,
            "fillOpacity": 0.6 if cluster != -1 else 0.1,
        }

    tooltip = GeoJsonTooltip(fields=["Neighborhood", "cluster"],
                             aliases=["Neighborhood:", "Cluster:"],
                             localize=True)

    layer = folium.FeatureGroup(name=f"{time_bin} Clusters")
    folium.GeoJson(merged, style_function=style_fn, tooltip=tooltip).add_to(layer)
    layer.add_to(m)

folium.LayerControl().add_to(m)
m.save(output_map_path)
print(f"✅ Map saved to {output_map_path}")
