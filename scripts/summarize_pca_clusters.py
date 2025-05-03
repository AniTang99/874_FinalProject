
import pandas as pd

# === Load Data ===
df = pd.read_csv("data/processed/merged_spd_weather.csv", low_memory=False)
df['Neighborhood'] = df['Dispatch Neighborhood'].astype(str).str.lower().str.strip()
df['Initial Call Type'] = df['Initial Call Type'].astype(str).str.lower().str.strip()

pca_clusters = pd.read_csv("output/neighborhood_pca_clusters.csv")
pca_clusters["Neighborhood"] = pca_clusters["Neighborhood"].str.lower().str.strip()

# === Merge to Associate Cluster Labels ===
df = df.merge(pca_clusters[["Neighborhood", "pca_cluster"]], on="Neighborhood", how="left")
df = df[df["pca_cluster"].notna()]

# === Generate Summary ===
summary = []
for cluster in sorted(df["pca_cluster"].unique()):
    cluster_df = df[df["pca_cluster"] == cluster]
    neighborhood_count = cluster_df["Neighborhood"].nunique()
    mean_calls = cluster_df.groupby("Neighborhood").size().mean()

    # Top 3 call types
    top_calls = (
        cluster_df["Initial Call Type"]
        .value_counts()
        .head(3)
        .index
        .tolist()
    )
    summary.append({
        "Cluster": int(cluster),
        "Neighborhoods": neighborhood_count,
        "Avg Calls per Neighborhood": round(mean_calls, 2),
        "Top Call Types": ", ".join(top_calls)
    })

summary_df = pd.DataFrame(summary)
summary_df.to_csv("output/pca_cluster_summary.csv", index=False)
print("✅ PCA cluster summary saved to output/pca_cluster_summary.csv")

# === Print Markdown Table ===
print("\n### PCA Cluster Summary (Markdown Table)\n")
print(summary_df.to_markdown(index=False))
