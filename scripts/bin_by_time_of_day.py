import pandas as pd
import os

# Load the dataset
input_path = "data/processed/merged_spd_weather.csv"
df = pd.read_csv(input_path, low_memory=False)

# Ensure datetime parsing
df["Queued Datetime"] = pd.to_datetime(df["CAD Event Original Time Queued"], errors="coerce")

# Drop rows with invalid datetime
df = df.dropna(subset=["Queued Datetime"])

# Define updated time binning function
def assign_time_bin(hour):
    if 0 <= hour < 4:
        return "Late Night"
    elif 4 <= hour < 6:
        return "Early Morning"
    elif 6 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 17:
        return "Afternoon"
    elif 17 <= hour < 21:
        return "Evening"
    else:
        return "Night"

# Apply binning
df["Hour"] = df["Queued Datetime"].dt.hour
df["Time of Day Bin"] = df["Hour"].apply(assign_time_bin)

# Save output to expected filename
os.makedirs("data/processed", exist_ok=True)
output_path = "data/processed/merged_spd_weather_timebins.csv"
df.to_csv(output_path, index=False)

print(f"✅ Updated time-of-day binning complete. Saved to {output_path}")
print("⏰ Time Bin counts:")
print(df["Time of Day Bin"].value_counts())
