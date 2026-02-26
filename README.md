# Credit Card Fraud Detection – Cost-Sensitive ML Modeling

## 📌 Project Overview
This project focuses on detecting fraudulent credit card transactions in a highly imbalanced dataset (0.17% fraud cases).

The objective is not just classification accuracy, but minimizing financial loss using cost-sensitive evaluation.

---

## ⚙️ Models Used
- Logistic Regression (class_weight="balanced")
- Random Forest (class_weight="balanced")
- SMOTE + Logistic Regression
- SMOTE + Random Forest

---

## 📊 Evaluation Metrics
Due to extreme class imbalance, the following metrics were prioritized:

- PR-AUC (Primary Metric)
- ROC-AUC
- Recall
- Precision
- F1 Score
- Cost-based evaluation

---

## 💡 Key Insights
- ROC-AUC alone is misleading for imbalanced data.
- PR-AUC provides better performance understanding.
- Threshold tuning significantly improves recall.
- Cost-sensitive evaluation changed the final deployment decision.
- Logistic Regression was selected for deployment due to lowest financial loss and better interpretability.

---

## 🏆 Final Model
Logistic Regression (class_weight="balanced")

---

## 📂 Files
- FRD project.ipynb – Complete modeling workflow

---

## 🚀 Future Improvements
- Advanced feature engineering
- XGBoost / LightGBM experimentation
- Deployment as API
