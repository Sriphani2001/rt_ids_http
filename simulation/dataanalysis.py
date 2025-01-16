import pandas as pd
import numpy as np
import os

# Load your dataset
file_path = "testttt.csv"  # Replace with the path to your dataset
data = pd.read_csv(file_path)

# Ensure data is sorted by time for accurate feature computation
data.sort_values(by="Time", inplace=True)

# Fill missing values
data.fillna(0, inplace=True)

# Define the output directory
output_dir = "unsuper_data/pre_logs"
os.makedirs(output_dir, exist_ok=True)

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

# Split data into chunks of 10,000 rows
chunk_size = 10000
total_chunks = len(data) // chunk_size + (1 if len(data) % chunk_size != 0 else 0)

for i in range(total_chunks):
    # Extract chunk
    chunk = data.iloc[i * chunk_size: (i + 1) * chunk_size]
    
    # Group by flow
    grouped = chunk.groupby(["Source", "Destination", "Protocol", "srcPort", "dstPort"])
    
    # Extract features for each flow
    flow_features = [calculate_flow_features(group) for _, group in grouped]
    
    # Convert to DataFrame
    flow_features_df = pd.DataFrame(flow_features)
    
    # Save to CSV
    output_csv = os.path.join(output_dir, f"pre_log({i + 1}).csv")
    flow_features_df.to_csv(output_csv, index=False)
    print(f"Chunk {i + 1}/{total_chunks} saved to {output_csv}")

print("All chunks processed and saved.")
