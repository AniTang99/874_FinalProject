import pandas as pd
import folium
import geopandas as gpd

# === Paths ===
csv_path = "output/neighborhood_kmeans_clusters.csv"  # Update if needed
geojson_path = "data/raw/spd_dispatch_neighborhoods.geojson"  # Update if needed
output_map_path = "output/kmeans_2_cluster_map.html"

# === Load Data ===
df = pd.read_csv(csv_path)
geojson = gpd.read_file(geojson_path)

# Normalize names
df["Neighborhood"] = df["Neighborhood"].str.strip().str.lower()
geojson["neighborhood"] = geojson["neighborhood"].str.strip().str.lower()

# Merge on lowercase neighborhood names
merged = geojson.merge(df, left_on="neighborhood", right_on="Neighborhood")

# Create map
m = folium.Map(location=[47.6062, -122.3321], zoom_start=11, tiles="cartodbpositron")

# Choropleth layer
folium.Choropleth(
    geo_data=merged,
    data=merged,
    columns=["neighborhood", "kmeans_cluster"],
    key_on="feature.properties.neighborhood",
    fill_color="Set1",
    fill_opacity=0.75,
    line_opacity=0.2,
    legend_name="KMeans Cluster (k=2)"
).add_to(m)

# Tooltip
folium.GeoJson(
    merged,
    tooltip=folium.GeoJsonTooltip(fields=["neighborhood", "kmeans_cluster"],
                                   aliases=["Neighborhood:", "Cluster:"])
).add_to(m)

# Save map
m.save(output_map_path)
print(f"✅ Map saved to: {output_map_path}")
