
import pandas as pd
import os
import matplotlib.pyplot as plt

# === Configuration ===
input_dir = "output/time_bin_clustering"
output_csv = os.path.join(input_dir, "time_bin_cluster_assignments.csv")
output_md = os.path.join(input_dir, "time_bin_cluster_summary.md")
output_plot = os.path.join(input_dir, "cluster_distribution_summary.png")

# === Load and Combine Cluster Files ===
time_bins = ["late_night", "early_morning", "morning", "afternoon", "evening", "night"]
combined_df = []

for bin_name in time_bins:
    file_path = os.path.join(input_dir, f"{bin_name}_clusters.csv")
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df["Time Bin"] = bin_name.replace("_", " ").title()
        if "Dispatch Neighborhood" in df.columns:
            df = df.rename(columns={"Dispatch Neighborhood": "Neighborhood"})
        combined_df.append(df)

if not combined_df:
    raise FileNotFoundError("❌ No cluster CSV files found in the specified directory.")

full_df = pd.concat(combined_df, ignore_index=True)
full_df.to_csv(output_csv, index=False)
print(f"✅ Combined cluster data saved to: {output_csv}")

# === Create Summary Table ===
summary = (
    full_df.groupby(["Time Bin", "cluster"])
    .agg(Neighborhoods=("Neighborhood", "count"))
    .reset_index()
    .sort_values(by=["Time Bin", "cluster"])
)

# === Save Summary to Markdown ===
with open(output_md, "w") as f:
    f.write("### Time Bin Cluster Summary\n\n")
    f.write(summary.to_markdown(index=False))
print(f"📄 Markdown summary saved to: {output_md}")

# === Save Plot of Distribution ===
plt.figure(figsize=(10, 6))
summary_pivot = summary.pivot(index="Time Bin", columns="cluster", values="Neighborhoods").fillna(0)
summary_pivot.plot(kind="bar", stacked=True, figsize=(12, 7), colormap="tab20")
plt.title("Neighborhood Distribution per Cluster Across Time Bins")
plt.xlabel("Time Bin")
plt.ylabel("Number of Neighborhoods")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(output_plot)
plt.close()
print(f"📊 Cluster distribution plot saved to: {output_plot}")
