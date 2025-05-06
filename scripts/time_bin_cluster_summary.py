import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

# === Paths ===
input_data_path = "output/merged_with_time_bins.csv"
cluster_assignments_path = "output/time_bin_clustering/time_bin_cluster_assignments.csv"
output_dir = "output/temporal_analysis"
os.makedirs(output_dir, exist_ok=True)

# === Load Data ===
df = pd.read_csv(input_data_path, low_memory=False)
clusters = pd.read_csv(cluster_assignments_path)

# === Normalize Column Names for Merge Compatibility ===
if "Neighborhood" in df.columns and "Dispatch Neighborhood" not in df.columns:
    df.rename(columns={"Neighborhood": "Dispatch Neighborhood"}, inplace=True)
if "Time Bin" in df.columns and "Time of Day Bin" not in df.columns:
    df.rename(columns={"Time Bin": "Time of Day Bin"}, inplace=True)

if "Time Bin" in clusters.columns and "Time of Day Bin" not in clusters.columns:
    clusters.rename(columns={"Time Bin": "Time of Day Bin"}, inplace=True)
if "Neighborhood" in clusters.columns and "Dispatch Neighborhood" not in clusters.columns:
    clusters.rename(columns={"Neighborhood": "Dispatch Neighborhood"}, inplace=True)

# === Merge Data ===
merged = df.merge(clusters, on=["Dispatch Neighborhood", "Time of Day Bin"], how="inner")

# === Profile Clusters by Time Bin ===
summary_list = []

for time_bin, group in merged.groupby("Time of Day Bin"):
    cluster_summary = group.groupby("cluster").agg(
        Call_Volume=("Initial Call Type", "count"),
        Avg_Priority=("Priority", "mean")
    ).reset_index()
    cluster_summary["Time Bin"] = time_bin

    top_calls = group.groupby(["cluster", "Initial Call Type"]).size().reset_index(name="count")
    top_calls = top_calls.sort_values(["cluster", "count"], ascending=[True, False])
    top_call_types = top_calls.groupby("cluster").head(3).groupby("cluster")["Initial Call Type"].apply(lambda x: ", ".join(x)).to_dict()
    cluster_summary["Top Call Types"] = cluster_summary["cluster"].map(top_call_types)

    summary_list.append(cluster_summary)

summary_df = pd.concat(summary_list)
summary_df.to_csv(os.path.join(output_dir, "time_bin_cluster_profile.csv"), index=False)

# === Cluster Transition Matrix ===
pivot = clusters.pivot(index="Dispatch Neighborhood", columns="Time of Day Bin", values="cluster")
pivot.to_csv(os.path.join(output_dir, "time_bin_cluster_transitions.csv"))

# === Save Markdown Summary ===
markdown_path = os.path.join(output_dir, "time_bin_cluster_profile.md")
with open(markdown_path, "w") as f:
    f.write("### Time Bin Cluster Profiles\n\n")
    f.write(summary_df.to_markdown(index=False))

print("✅ Cluster profiling and transition matrix generated successfully.")
