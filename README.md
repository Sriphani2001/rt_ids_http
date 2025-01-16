# Intrusion Detection System (IDS) for HTTP Flood Attacks Using ML 

## Overview
I built an IDS to detect HTTP flood attacks. These attacks disrupt services by overwhelming servers with HTTP requests. My system uses ML and DL models for detection. I also created a simulation pipeline for continuous testing.

---

## Aim
To create an IDS for HTTP flood detection using trained models and test new preprocessed data in a simulation pipeline.

---

## Dataset
I used the **CIC 2017 DDoS Friday Afternoon Dataset**, which includes labeled normal and DDoS attack traffic.
- **Training Data:** Subset used for model training.
- **Testing Data:** Separate and additional preprocessed datasets for testing.

Key features:
- Network ports
- Packet sizes
- Traffic types

---

## Methodology
### 1. Data Preprocessing
- Cleaned and formatted raw data.
- Split data into training and testing sets.
- Selected relevant features.

### 2. Models
- **ML Algorithms:** SVM, kNN, Linear Regression.


Testing models:
```json
{
  "SVM": "../model/svm_model.pkl",
  "LR": "../model/log_reg_model.pkl",
  "KNN": "../model/KNN_model.pkl"
}
```

### 3. Explainability
- Used SHAP to explain model predictions and identify critical features.

### 4. Simulation and Testing
- **Pipeline:** Tested new preprocessed data for flexibility.
- **SNORT Integration:** Captured real-time traffic for testing and dataset creation.

### 5. Evaluation
- Metrics: Accuracy, Precision, Recall, F1-Score.
- Compared all models to find the best.

---

## Results
- SVM and CNN had the best accuracy.
- SHAP explained predictions clearly.
- Simulation pipeline and SNORT worked well in real-time tests.

---

## Future Work
- Add detection for SYN flood attacks.
- Use advanced anomaly detection.
- Optimize DL models for faster detection.

---

## How to Use
1. Train models using the CIC 2017 dataset.
2. Test new preprocessed data in the simulation pipeline.
3. Use SNORT to monitor real-time traffic.

---

## Repository Structure
- `data/`: CIC 2017 dataset and preprocessed data.
- `models/`: Trained models.
- `scripts/`: Code for preprocessing, training, and testing.
- `results/`: Evaluation metrics and SHAP outputs.
- `pipeline/`: Simulation pipeline.

---

## Acknowledgments
Thanks to CIC 2017 creators and SNORT for enabling this project.

