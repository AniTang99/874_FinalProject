import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os

# === Load Data ===
df = pd.read_csv("output/neighborhood_feature_matrix.csv")

# === Ensure output folder exists ===
os.makedirs("output", exist_ok=True)

# === Check for necessary columns ===
if "Neighborhood" not in df.columns or "call_count" not in df.columns:
    raise KeyError("Missing required columns: 'Neighborhood' and/or 'call_count'")

# === Prepare features ===
X = df.select_dtypes(include=["number"])
neighborhoods = df["Neighborhood"].values

# === Drop rows with missing values ===
valid_rows = X.dropna()
neighborhoods_clean = neighborhoods[valid_rows.index]
X_clean = valid_rows.values

print(f"🔍 Valid samples remaining after NaN removal: {len(X_clean)} of {len(X)}")

# === KMeans Clustering with Silhouette Scores ===
range_n_clusters = range(2, 10)
scores = []
models = []

for k in range_n_clusters:
    model = KMeans(n_clusters=k, random_state=42)
    labels = model.fit_predict(X_clean)
    score = silhouette_score(X_clean, labels)
    scores.append(score)
    models.append((model, labels))

# === Plot Silhouette Scores ===
plt.figure(figsize=(8, 6))
plt.plot(range_n_clusters, scores, marker="o")
plt.title("Silhouette Scores for K-Means Clustering")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.grid(True)
plt.tight_layout()
plt.savefig("output/kmeans_silhouette_plot.png")
plt.close()

# === Best Model ===
best_index = np.argmax(scores)
best_k = range_n_clusters[best_index]
best_model, best_labels = models[best_index]

# === Save Cluster Assignments ===
cluster_df = pd.DataFrame({
    "Neighborhood": neighborhoods_clean,
    "kmeans_cluster": best_labels
})
cluster_df.to_csv("output/neighborhood_kmeans_clusters.csv", index=False)

# === Merge cluster info back into feature matrix ===
df_full = df.merge(cluster_df, on="Neighborhood", how="left")

# === Cluster Summary ===
top_calls = df_full.groupby(["kmeans_cluster", "Top Call Type 1"]).size().reset_index(name="count")
top_calls = top_calls.sort_values(["kmeans_cluster", "count"], ascending=[True, False])

summary = (
    df_full.groupby("kmeans_cluster")
    .agg(Neighborhoods=("Neighborhood", "count"), Avg_Calls=("call_count", "mean"))
    .reset_index()
)

top_call_types = top_calls.groupby("kmeans_cluster").head(3)
top_call_map = (
    top_call_types.groupby("kmeans_cluster")["Top Call Type 1"]
    .apply(lambda x: ", ".join(x))
    .to_dict()
)
summary["Top Call Types"] = summary["kmeans_cluster"].map(top_call_map)

summary.to_csv("output/kmeans_cluster_summary.csv", index=False)

with open("output/kmeans_cluster_summary.md", "w") as f:
    f.write("### K-Means Cluster Summary (Markdown Table)\n\n")
    f.write(summary.to_markdown(index=False))

# === Cluster Distribution Plot ===
plt.figure(figsize=(8, 6))
sns.countplot(data=cluster_df, x="kmeans_cluster", palette="Set2")
plt.title("Number of Neighborhoods per Cluster")
plt.xlabel("Cluster")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("output/kmeans_cluster_distribution.png")
plt.close()

print("✅ Analysis complete. Plots and summaries saved to 'output/'.")
