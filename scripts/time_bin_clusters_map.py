
import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import TimestampedGeoJson
import os

# Paths
geojson_path = "data/raw/spd_dispatch_neighborhoods.geojson"
clusters_folder = "output/time_bin_clustering"
output_path = "output/time_bin_clustering/time_bin_clusters_map.html"

# Time bins
time_bins = ["Late Night", "Early Morning", "Morning", "Afternoon", "Evening", "Night"]
bin_file_map = {
    bin: f"{clusters_folder}/{bin.lower().replace(' ', '_')}_clusters.csv" for bin in time_bins
}

# Load GeoJSON
gdf = gpd.read_file(geojson_path)
gdf = gdf.rename(columns={"neighborhood": "Neighborhood"})
gdf["Neighborhood"] = gdf["Neighborhood"].str.lower().str.strip()
gdf = gdf.to_crs(epsg=4326)

# Prepare layers for each time bin
features = []

for time_bin, csv_path in bin_file_map.items():
    if not os.path.exists(csv_path):
        print(f"⛔ Skipping {time_bin} — file not found: {csv_path}")
        continue

    df = pd.read_csv(csv_path)
    df["Dispatch Neighborhood"] = df["Dispatch Neighborhood"].astype(str).str.lower().str.strip()

    # Merge with GeoJSON
    merged = gdf.merge(df, left_on="Neighborhood", right_on="Dispatch Neighborhood", how="inner")
    timestamp = f"{time_bins.index(time_bin)+1:02d}-2025"

    for _, row in merged.iterrows():
        feature = {
            "type": "Feature",
            "geometry": row["geometry"].__geo_interface__,
            "properties": {
                "time": timestamp,
                "style": {
                    "color": "black",
                    "weight": 1,
                    "fillOpacity": 0.7
                },
                "icon": "circle",
                "popup": f"{row['Dispatch Neighborhood'].title()}<br>Cluster: {row['cluster']}"
            }
        }
        features.append(feature)

# Create map
m = folium.Map(location=[47.6, -122.33], zoom_start=11, tiles="CartoDB positron")

TimestampedGeoJson({
    "type": "FeatureCollection",
    "features": features,
}, period="P1M", add_last_point=False, auto_play=False, loop=False, max_speed=1,
   loop_button=True, date_options="YYYY", time_slider_drag_update=True).add_to(m)

# Save
os.makedirs(os.path.dirname(output_path), exist_ok=True)
m.save(output_path)
print(f"✅ Saved to {output_path}")
