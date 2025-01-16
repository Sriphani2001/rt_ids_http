import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the validation data from extracted_features .csv, unsuper_data/test_logs/test_log(1)
validation_file_path = "data/X_val.csv"

# Ensure the file exists
if not os.path.exists(validation_file_path):
    raise FileNotFoundError(f"Validation file not found at {validation_file_path}. Please ensure it exists.")

# Load the validation data
validation_data = pd.read_csv(validation_file_path)

# Scale the features directly using MinMaxScaler
scaler = MinMaxScaler()
X_val = validation_data.values  # Assuming all columns are features
X_val_scaled = scaler.fit_transform(X_val)

# Paths to your saved models
model_paths = {
    "KNN":"model/KNN_model.pkl",
    "SVM": "model/svm_model.pkl",
    "LR": "model/log_reg_model.pkl",
}

# Initialize predictions
predictions = {
    "Model": [],
    "Predicted Probabilities": [],
    "Threshold-Based Predictions": [],
}

# Set a decision threshold (e.g., 0.5)
decision_threshold = 0.5

# Evaluate each model
for model_name, model_path in model_paths.items():
    try:
        # Load the model
        with open(model_path, 'rb') as file:
            model = pickle.load(file)

        # Predict probabilities
        if hasattr(model, "decision_function"):  # For models like SVM
            y_probs = model.decision_function(X_val_scaled)
        elif hasattr(model, "predict_proba"):  # For models like Logistic Regression
            y_probs = model.predict_proba(X_val_scaled)[:, 1]
        elif hasattr(model, "predict"):  # For models like neural networks
            y_probs = model.predict(X_val_scaled).ravel()  # Ensure 1D array
        else:
            raise AttributeError(f"{model_name} does not support probability prediction.")

        # Threshold-based predictions
        y_preds = (y_probs > decision_threshold).astype(int)

        # Store predictions
        predictions["Model"].append(model_name)
        predictions["Predicted Probabilities"].append(y_probs)
        predictions["Threshold-Based Predictions"].append(y_preds)

        # Visualize the predicted probabilities
        plt.figure(figsize=(10, 6))
        sns.histplot(y_probs, bins=20, kde=True, color="blue")
        plt.axvline(decision_threshold, color="red", linestyle="--", label="Decision Threshold")
        plt.title(f"Predicted Probabilities for {model_name}")
        plt.xlabel("Predicted Probability")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()

    except Exception as e:
        print(f"An error occurred with {model_name}: {e}")

# Print predicted probabilities and decisions
print("\nPredictions for Each Model:")
for i, model in enumerate(predictions["Model"]):
    print(f"Model: {model}")
    print(f"Predicted Probabilities: {predictions['Predicted Probabilities'][i]}")
    print(f"Threshold-Based Predictions: {predictions['Threshold-Based Predictions'][i]}")
    print("\n")

# Save predictions to CSV
output_csv_path = "data/model_predictions.csv"
os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)  # Ensure directory exists
predictions_df = pd.DataFrame({
    "Model": predictions["Model"],
    "Predicted Probabilities": [";".join(map(str, probs)) for probs in predictions["Predicted Probabilities"]],
    "Threshold-Based Predictions": [";".join(map(str, preds)) for preds in predictions["Threshold-Based Predictions"]],
})
predictions_df.to_csv(output_csv_path, index=False)
print(f"Predictions saved to {output_csv_path}")
