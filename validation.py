import pandas as pd
import os
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.exceptions import NotFittedError
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load the validation data from processed_data.csv
validation_file_path = "data/validation.csv"

# Ensure the file exists
if not os.path.exists(validation_file_path):
    raise FileNotFoundError(f"Validation file not found at {validation_file_path}. Please ensure it exists.")

# Load the validation data
validation_data = pd.read_csv(validation_file_path)

# Separate features (X_val) and labels (Y_val)
X_val = validation_data.drop(columns=["Label_cleaned"]).values  # Features
Y_val = validation_data["Label_cleaned"].values  # Labels

# Apply Min-Max scaling to the features
scaler = MinMaxScaler()
X_val_scaled = scaler.fit_transform(X_val)

# Paths to your saved models
model_paths = {
    "KNN":"model/KNN_model.pkl",
    "SVM": "model/svm_model.pkl",
    "LR": "model/log_reg_model.pkl",
    "LGB": "model/lgb_model.pkl",
  
}
# Initialize metrics
performance_metrics = {
    "Model": [],
    "Detected Attacks (TP)": [],
    "Detected Non-Attacks (TN)": [],
    "False Positives (FP)": [],
    "False Negatives (FN)": [],
    "Accuracy (%)": [],
    "Precision": [],
    "Recall": [],
    "F1-Score": [],
    "ROC-AUC": [],
}

# Evaluate each model
for model_name, model_path in model_paths.items():
    try:
        # Load the model
        with open(model_path, 'rb') as file:
            model = pickle.load(file)

        # Predict class labels
        y_preds = model.predict(X_val_scaled)

        # Check if the model has predict_proba for ROC-AUC calculation
        if hasattr(model, "predict_proba"):
            y_probs = model.predict_proba(X_val_scaled)[:, 1]
            roc_auc = roc_auc_score(Y_val, y_probs)
        else:
            roc_auc = "N/A"  # For models that don't support predict_proba

        # Calculate confusion matrix components
        tn, fp, fn, tp = confusion_matrix(Y_val, y_preds).ravel()

        # Calculate metrics
        accuracy = accuracy_score(Y_val, y_preds) * 100
        precision = precision_score(Y_val, y_preds, zero_division=0)
        recall = recall_score(Y_val, y_preds)
        f1 = f1_score(Y_val, y_preds)

        # Store metrics
        performance_metrics["Model"].append(model_name)
        performance_metrics["Detected Attacks (TP)"].append(tp)
        performance_metrics["Detected Non-Attacks (TN)"].append(tn)
        performance_metrics["False Positives (FP)"].append(fp)
        performance_metrics["False Negatives (FN)"].append(fn)
        performance_metrics["Accuracy (%)"].append(accuracy)
        performance_metrics["Precision"].append(precision)
        performance_metrics["Recall"].append(recall)
        performance_metrics["F1-Score"].append(f1)
        performance_metrics["ROC-AUC"].append(roc_auc)

        # Confusion Matrix Visualization
        cm = confusion_matrix(Y_val, y_preds)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Non-Attack", "Attack"], yticklabels=["Non-Attack", "Attack"])
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"Confusion Matrix for {model_name}")
        plt.show()

    except NotFittedError:
        print(f"Model {model_name} is not fitted. Please train it first.")
    except Exception as e:
        print(f"An error occurred with {model_name}: {e}")

# Convert performance metrics to a DataFrame for better visualization
performance_df = pd.DataFrame(performance_metrics)

# Save performance metrics to CSV
output_csv_path = "data/model_performance_metrics.csv"
performance_df.to_csv(output_csv_path, index=False)
print(f"Performance metrics saved to {output_csv_path}")

# Print metrics for each model
print("Model Performance Comparison:")
print(performance_df)

# Bar chart for additional comparison metrics
plt.figure(figsize=(10, 7))
x = np.arange(len(performance_df["Model"]))
width = 0.2

# Bar positions for various metrics
bars_tp = plt.bar(x - width, performance_df["Detected Attacks (TP)"], width, label="Detected Attacks (TP)")
bars_tn = plt.bar(x, performance_df["Detected Non-Attacks (TN)"], width, label="Detected Non-Attacks (TN)")
bars_fp = plt.bar(x + width, performance_df["False Positives (FP)"], width, label="False Positives (FP)")

# Add bar labels
plt.bar_label(bars_tp, padding=3)
plt.bar_label(bars_tn, padding=3)
plt.bar_label(bars_fp, padding=3)

# Plot settings
plt.xlabel("Models")
plt.ylabel("Number of Instances")
plt.title("Performance Comparison: TP, TN, and FP")
plt.xticks(x, performance_df["Model"])
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Display Precision, Recall, F1-Score, and ROC-AUC in a separate chart
plt.figure(figsize=(10, 7))
plt.plot(performance_df["Model"], performance_df["Precision"], marker="o", label="Precision")
plt.plot(performance_df["Model"], performance_df["Recall"], marker="o", label="Recall")
plt.plot(performance_df["Model"], performance_df["F1-Score"], marker="o", label="F1-Score")

if "ROC-AUC" in performance_df.columns and performance_df["ROC-AUC"].dtype != "object":
    plt.plot(performance_df["Model"], performance_df["ROC-AUC"], marker="o", label="ROC-AUC")

# Plot settings for metrics
plt.xlabel("Models")
plt.ylabel("Metric Values")
plt.title("Precision, Recall, F1-Score, and ROC-AUC for Models")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
