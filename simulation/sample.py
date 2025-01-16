import pandas as pd
import numpy as np

# Load your dataset
file_path = "extracted_data.csv"  # Replace with the path to your dataset
data = pd.read_csv(file_path)

# Load benign dataset
benign_file_path = "data/benign.csv"  # Path to the benign dataset
benign_data = pd.read_csv(benign_file_path)

# Ensure data is sorted by time for accurate feature computation

benign_data.sort_values(by="Time", inplace=True)

# Fill missing values
data.fillna(0, inplace=True)
benign_data.fillna(0, inplace=True)

# Reduce to the first 2500 rows
data = data.head(2500)

# Function to calculate features for each flow
def calculate_flow_features(group):
    features = {}

    # Flow Duration
    features["Flow Duration"] = group["Time"].max() - group["Time"].min()

    # Total Fwd Packets
    features["Total Fwd Packets"] = len(group)

    # Total Length of Fwd Packets
    features["Total Length of Fwd Packets"] = group["Length"].sum()

    # Fwd Packet Length Max
    features["Fwd Packet Length Max"] = group["Length"].max()

    # Fwd Packet Length Min
    features["Fwd Packet Length Min"] = group["Length"].min()

    # Fwd Packet Length Mean
    features["Fwd Packet Length Mean"] = group["Length"].mean()

    # Bwd Packet Length Max
    features["Bwd Packet Length Max"] = group["Length"].max()

    # Bwd Packet Length Min
    features["Bwd Packet Length Min"] = group["Length"].min()

    # Flow Bytes/s
    duration = features["Flow Duration"]
    features["Flow Bytes/s"] = features["Total Length of Fwd Packets"] / duration if duration > 0 else 0

    # Flow Packets/s
    features["Flow Packets/s"] = features["Total Fwd Packets"] / duration if duration > 0 else 0

    # Flow IAT Mean and Min
    inter_arrival_times = group["Time"].diff().dropna()
    features["Flow IAT Mean"] = inter_arrival_times.mean()
    features["Flow IAT Min"] = inter_arrival_times.min()

    # Fwd IAT Min
    features["Fwd IAT Min"] = inter_arrival_times.min()

    # Bwd IAT Mean
    features["Bwd IAT Mean"] = inter_arrival_times.mean()

    # Fwd PSH Flags
    features["Fwd PSH Flags"] = group["Info"].str.contains("PSH", na=False).sum()

    # Bwd Packets/s
    features["Bwd Packets/s"] = features["Total Fwd Packets"] / duration if duration > 0 else 0

    # Minimum Packet Length
    features["Min Packet Length"] = group["Length"].min()

    # SYN Flag Count
    features["SYN Flag Count"] = group["Info"].str.contains("SYN", na=False).sum()

    # ACK Flag Count
    features["ACK Flag Count"] = group["Info"].str.contains("ACK", na=False).sum()

    # URG Flag Count
    features["URG Flag Count"] = group["Info"].str.contains("URG", na=False).sum()

    # Down/Up Ratio
    features["Down/Up Ratio"] = features["Bwd Packet Length Max"] / features["Fwd Packet Length Min"] if features["Fwd Packet Length Min"] > 0 else 0

    # Initial Window Sizes
    win_sizes = group["Info"].str.extract(r"Win=(\d+)").dropna().astype(float)
    features["Init_Win_bytes_forward"] = win_sizes.mean().iloc[0] if not win_sizes.empty else 0
    features["Init_Win_bytes_backward"] = win_sizes.mean().iloc[0] if not win_sizes.empty else 0

    # Active Mean and Std
    features["Active Mean"] = inter_arrival_times.mean() if len(inter_arrival_times) > 0 else 0
    features["Active Std"] = inter_arrival_times.std() if len(inter_arrival_times) > 0 else 0

    # act_data_pkt_fwd
    features["act_data_pkt_fwd"] = group[group["Length"] > 0].shape[0]

    # min_seg_size_forward
    if "Segment Size" in group.columns:
        features["min_seg_size_forward"] = group["Segment Size"].min()
    else:
        features["min_seg_size_forward"] = 0  # Default if the column is missing

    return features

# Group by flow and calculate features for data
grouped = data.groupby(["Source", "Destination", "Protocol", "srcPort", "dstPort"])
flow_features = [calculate_flow_features(group) for _, group in grouped]

# Convert to DataFrame
flow_features_df = pd.DataFrame(flow_features)

# Combine with benign dataset
combined_data = pd.concat([flow_features_df, benign_data], ignore_index=True)

# Save the combined dataset to a new file
combined_file_path = "extracted_features.csv"
combined_data.to_csv(combined_file_path, index=False)

print(f"Combined dataset saved to {combined_file_path}")
