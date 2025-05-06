
import pandas as pd
import os

# Load merged data
df = pd.read_csv("data/processed/merged_spd_weather.csv", low_memory=False)

# Standardize neighborhood field
df['Neighborhood'] = df['Dispatch Neighborhood'].astype(str).str.lower().str.strip()
df = df[~df['Neighborhood'].isin(['-', 'unknown']) & df['Neighborhood'].notna()]

# Compute call counts
call_counts = df.groupby("Neighborhood").size().rename("call_count")

# Check if 'Priority' column exists instead of 'Initial Call Priority'
if "Priority" in df.columns:
    priority_avg = df.groupby("Neighborhood")["Priority"].apply(
        lambda x: pd.to_numeric(x, errors="coerce").mean()
    ).rename("avg_priority")
else:
    print("⚠️ 'Priority' not found in data. Skipping priority averaging.")
    priority_avg = pd.Series(index=call_counts.index, data=[None] * len(call_counts), name="avg_priority")

# Extract most common call type
top_calls = (
    df.groupby(["Neighborhood", "Initial Call Type"])
    .size()
    .reset_index(name="count")
    .sort_values(["Neighborhood", "count"], ascending=[True, False])
)
top_types = top_calls.groupby("Neighborhood").head(1)[["Neighborhood", "Initial Call Type"]]
top_types = top_types.rename(columns={"Initial Call Type": "Top Call Type 1"})

# Merge features
feature_df = pd.DataFrame(call_counts).join(priority_avg).reset_index()
feature_df = feature_df.merge(top_types, on="Neighborhood", how="left")

# Output
os.makedirs("output", exist_ok=True)
feature_df.to_csv("output/neighborhood_feature_matrix.csv", index=False)
print("✅ Saved to output/neighborhood_feature_matrix.csv")
