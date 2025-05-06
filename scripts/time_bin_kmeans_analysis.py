
import pandas as pd
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the time-binned dataset
df = pd.read_csv("data/processed/merged_spd_weather_timebins.csv", low_memory=False)

# Filter out unknown or placeholder neighborhoods
df = df[~df["Dispatch Neighborhood"].isin(["-", "unknown"])]

# Ensure output directory exists
os.makedirs("output/time_bin_clustering", exist_ok=True)

# Define bins to iterate
bins = ["Late Night", "Early Morning", "Morning", "Afternoon", "Evening", "Night"]

# Collect silhouette scores for comparison
bin_summaries = []

for time_bin in bins:
    bin_df = df[df["Time of Day Bin"] == time_bin]
    
    # Compute basic features for clustering
    grouped = (
        bin_df.groupby("Dispatch Neighborhood")
        .agg(call_count=("Initial Call Type", "count"),
             avg_priority=("Priority", "mean"))
        .dropna()
    )

    if grouped.empty or len(grouped) < 2:
        print(f"⛔ Not enough data to cluster for {time_bin}")
        continue

    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(grouped)

    # Choose optimal number of clusters using silhouette scores
    best_score = -1
    best_k = 0
    best_labels = None

    for k in range(2, min(len(grouped), 10)):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels

    grouped["cluster"] = best_labels
    grouped.reset_index().to_csv(f"output/time_bin_clustering/{time_bin.lower().replace(' ', '_')}_clusters.csv", index=False)

    bin_summaries.append((time_bin, best_k, best_score))

    # Plot
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=best_labels, palette="tab10")
    plt.title(f"{time_bin} Clusters (k={best_k}, Silhouette: {best_score:.2f})")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend(title="Cluster")
    plt.tight_layout()
    plt.savefig(f"output/time_bin_clustering/{time_bin.lower().replace(' ', '_')}_plot.png")
    plt.close()

# Save summary
summary_df = pd.DataFrame(bin_summaries, columns=["Time Bin", "Best k", "Silhouette Score"])
summary_df.to_csv("output/time_bin_clustering/summary.csv", index=False)
print("✅ Time-aware clustering (with unknowns excluded) complete. Results in output/time_bin_clustering/")
