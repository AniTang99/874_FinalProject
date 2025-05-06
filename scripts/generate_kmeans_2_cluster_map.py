import pandas as pd
import geopandas as gpd
import folium
from shapely.geometry import mapping

# === Paths ===
csv_path = "output/neighborhood_kmeans_clusters.csv"  # Make sure this CSV exists
geojson_path = "data/raw/spd_dispatch_neighborhoods.geojson"  # Should contain the geometry
output_map_path = "output/kmeans_2_cluster_map.html"  # Output path for the HTML map

# === Load Data ===
df = pd.read_csv(csv_path)
geo = gpd.read_file(geojson_path)

# === Normalize Neighborhood Names for Merging ===
df["Neighborhood"] = df["Neighborhood"].str.strip().str.lower()
geo["neighborhood"] = geo["neighborhood"].str.strip().str.lower()

# === Merge ===
merged = geo.merge(df, left_on="neighborhood", right_on="Neighborhood")

if merged.empty:
    raise ValueError("❌ Merge failed: No matching neighborhoods found between cluster CSV and GeoJSON.")

# Filter to only 2-cluster result (if not already filtered)
if "kmeans_cluster" not in merged.columns:
    raise KeyError("❌ 'kmeans_cluster' column not found in input CSV.")

# === Create Map ===
center = [47.6062, -122.3321]  # Seattle center
m = folium.Map(location=center, zoom_start=11)

colors = ["red", "blue", "green", "purple", "orange"]

for _, row in merged.iterrows():
    cluster = int(row["kmeans_cluster"])
    color = colors[cluster % len(colors)]
    geo_json = mapping(row["geometry"])
    
    folium.GeoJson(
        geo_json,
        style_function=lambda feature, col=color: {
            "fillColor": col,
            "color": "black",
            "weight": 1,
            "fillOpacity": 0.6
        },
        tooltip=f"{row['neighborhood'].title()} (Cluster {cluster})"
    ).add_to(m)

# === Save Map ===
m.save(output_map_path)
print(f"✅ Map saved to {output_map_path}")
