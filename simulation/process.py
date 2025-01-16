import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import os
import pickle

# Define paths  unsuper_data/pre_logs/pre_log(1)
dataset_path = "extracted_features.csv"
preprocessed_data_path = "unsuper_data/test_logs/test_log(1).csv"

# Load the dataset
print("Loading dataset...")
data = pd.read_csv(dataset_path)

def remove_correlated_features(dataset, threshold=0.8):
    # Compute the correlation matrix
    corr_matrix = dataset.corr().abs()
    
    # Create a mask to identify the upper triangle of the matrix
    upper_tri = corr_matrix.where(
        ~np.tril(np.ones(corr_matrix.shape)).astype(bool)
    )
    
    # Find the indices of columns to drop
    to_drop = [
        column for column in upper_tri.columns
        if any(upper_tri[column] > threshold)
    ]
    
    # Drop the correlated columns
    dataset_cleaned = dataset.drop(columns=to_drop)
    
    return dataset_cleaned, to_drop

# List of columns to drop
columns_to_drop = [
    'Unnamed: 0',
    'Flow ID',
    'Src IP', 'Source IP',
    'Dst IP',
    'Source Port',
    'Destination IP',
    'Protocol',
    'Timestamp',
    'SimillarHTTP',
    'Inbound',
]

# Check for columns that exist in the dataset and drop them
columns_to_remove = list(set(columns_to_drop).intersection(set(data.columns)))
data = data.drop(columns=columns_to_remove, errors='ignore')

print("Removing duplicate rows...")
data = data.drop_duplicates()

# Strip column names of whitespace
data.columns = data.columns.str.strip()
import pandas as pd
import numpy as np  # Import NumPy directly

# Function to identify and remove correlated features
threshold = 0.8  # Set the threshold
# data, removed_features = remove_correlated_features(data, threshold)

# Output the results
# print("Removed Correlated Features:", removed_features)


# Clean missing values
data = data.dropna()



# Drop the original 'Label' and other unwanted columns
if "Destination Port" in data.columns:
    data = data.drop(columns=["Destination Port", "Label"], errors='ignore')

# Replace infinite values
print("Replacing infinite values...")
# data.replace([np.inf, -np.inf], np.nan, inplace=True)
# data.dropna(inplace=True)

###########
numeric_cols = data.select_dtypes(include=[np.number])

# Check for infinite values in numeric columns
inf_values_count = np.isinf(numeric_cols).sum().sum()
print(f"Number of infinite values in the numeric columns: {inf_values_count}")

# If there are infinite values, replace them with NaN
data.replace([np.inf, -np.inf], np.nan, inplace=True)

# Check for missing values
missing_values_count = data.isnull().sum().sum()
print(f"Number of missing values: {missing_values_count}")
# #####



# Filter out rows with infinities
data = data[~np.isinf(numeric_cols).any(axis=1)]

# Remove columns with a single unique value
data = data.loc[:, data.nunique() > 1]

#######


data = data.sample(frac=1, random_state=42).reset_index(drop=True)

data = data.loc[:, data.nunique() > 1]

# Apply Label Encoding to each non-numeric column
for column in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])




# Save the preprocessed data
print("Saving preprocessed data...")
os.makedirs(os.path.dirname(preprocessed_data_path), exist_ok=True)
data.to_csv(preprocessed_data_path, index=False)

print(f"Preprocessing complete. Preprocessed data saved to {preprocessed_data_path}")
