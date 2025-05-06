
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load data (ensure this file exists locally)
df = pd.read_csv("data/processed/merged_spd_weather_timebins.csv", low_memory=False)

# Ensure output directory exists
os.makedirs("output/temporal_analysis", exist_ok=True)

# Mapping for bin labels with time ranges
bin_label_map = {
    "Late Night": "Late Night (00–04)",
    "Early Morning": "Early Morning (04–06)",
    "Morning": "Morning (06–12)",
    "Afternoon": "Afternoon (12–17)",
    "Evening": "Evening (17–21)",
    "Night": "Night (21–00)"
}

# Apply new labels
df["Time Bin Label"] = df["Time of Day Bin"].map(bin_label_map)

# Plot call volume by time bin
plt.figure(figsize=(10, 6))
order = ['Late Night (00–04)', 'Early Morning (04–06)', 'Morning (06–12)',
         'Afternoon (12–17)', 'Evening (17–21)', 'Night (21–00)']
sns.countplot(data=df, x="Time Bin Label", order=order, palette="viridis")
plt.title("911 Call Volume by Time of Day")
plt.xlabel("Time of Day")
plt.ylabel("Number of Calls")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("output/temporal_analysis/call_volume_by_time.png")
plt.close()

# Plot call type distribution by time bin
top_call_types = df["Initial Call Type"].value_counts().head(10).index
df_filtered = df[df["Initial Call Type"].isin(top_call_types)]

plt.figure(figsize=(12, 8))
sns.countplot(data=df_filtered, y="Initial Call Type", hue="Time Bin Label", order=top_call_types, palette="viridis")
plt.title("Top 10 Call Types by Time of Day")
plt.xlabel("Number of Calls")
plt.ylabel("Initial Call Type")
plt.legend(title="Time Bin", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("output/temporal_analysis/call_type_by_time.png")
plt.close()

print("✅ Temporal visualizations saved to output/temporal_analysis/")
