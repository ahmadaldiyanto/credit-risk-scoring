# 💳 Credit Risk Scoring System

A Machine Learning web application to predict credit default risk using Tuned XGBoost and deployed with Streamlit.

---

## 📌 Project Overview

This project aims to build a Credit Risk Prediction model that classifies loan applications into:

- ✅ APPROVED (Low Risk)
- ❌ REJECTED (High Risk)

The model predicts the probability of default based on borrower financial information and loan details.

---

## 🧠 Machine Learning Model

- Algorithm: Tuned XGBoost Classifier
- Evaluation Metric: ROC-AUC
- Final Test ROC-AUC: 0.717
- Class Imbalance Handling: `scale_pos_weight`
- Feature Engineering:
  - loan_income_ratio
  - credit_exposure_ratio
  - debt_load_ratio
  - risk_interaction
  - One-hot encoding (state, purpose, home ownership, term)

The model was selected because:
- Highest ROC-AUC score
- Stable recall for bad loan detection
- Good generalization (small train-test gap)
- Suitable for tabular & imbalanced data

---

## 📊 Features Used

Main features include:

- Loan amount
- Annual income
- Interest rate
- Debt-to-income ratio
- Total accounts
- Open accounts
- Credit inquiries (last 6 months)
- Loan grade
- Loan purpose
- Home ownership
- State
- Engineered financial ratios

---

## 🚀 Deployment

This application is deployed using:

- Streamlit Cloud
- GitHub Repository Integration

To run locally:

```bash
pip install -r requirements.txt
streamlit run app.py