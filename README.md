<div align="center">

```
 ██████╗██╗  ██╗██╗   ██╗██████╗ ███╗   ██╗    ██████╗ ██████╗ ███████╗██████╗ ██╗ ██████╗████████╗
██╔════╝██║  ██║██║   ██║██╔══██╗████╗  ██║    ██╔══██╗██╔══██╗██╔════╝██╔══██╗██║██╔════╝╚══██╔══╝
██║     ███████║██║   ██║██████╔╝██╔██╗ ██║    ██████╔╝██████╔╝█████╗  ██║  ██║██║██║        ██║
██║     ██╔══██║██║   ██║██╔══██╗██║╚██╗██║    ██╔═══╝ ██╔══██╗██╔══╝  ██║  ██║██║██║        ██║
╚██████╗██║  ██║╚██████╔╝██║  ██║██║ ╚████║    ██║     ██║  ██║███████╗██████╔╝██║╚██████╗   ██║
 ╚═════╝╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝     ╚═╝  ╚═╝╚══════╝╚═════╝ ╚═╝ ╚═════╝   ╚═╝
```

### *Know Who's Leaving. Before They Leave.*

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-F7931E?style=for-the-badge)](https://xgboost.readthedocs.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

---

> **An end-to-end ML pipeline that predicts which e-commerce customers will churn — with 93% accuracy — deployed as a live Flask web app.**

</div>

---

## ◈ The Problem

Every lost customer is a revenue leak. This project builds a battle-tested churn prediction engine that identifies at-risk customers before they disappear — giving businesses the window to intervene.

```
┌────────────────────────────────────────────────────────────────┐
│                    CHURN PREDICTION FLOW                       │
│                                                                │
│  Customer Data  →  Feature Engineering  →  SMOTE Balancing    │
│       ↓                   ↓                      ↓            │
│  5,000 records      Tenure, Cashback,       16.8% churn →     │
│  16 features        Complaints, etc.        balanced set       │
│                           ↓                                    │
│         ┌────────────────────────────────┐                    │
│         │   Logistic Regression (~84%)   │                    │
│         │   Random Forest       (~91%)   │                    │
│         │ ✦ XGBoost             (~93%) ✦ │                    │
│         └────────────────────────────────┘                    │
│                           ↓                                    │
│               Flask Web App  +  REST API                       │
└────────────────────────────────────────────────────────────────┘
```

---

## ◈ Model Performance

| Model | Accuracy | F1 (Churn) | ROC-AUC |
|---|---|---|---|
| Logistic Regression | ~84% | ~0.60 | ~0.87 |
| Random Forest | ~91% | ~0.75 | ~0.96 |
| **XGBoost** ✦ | **~93%** | **~0.80** | **~0.97** |

> SMOTE boosted recall on the churn class by ~12 percentage points across all models.

---

## ◈ Key Intelligence

- **Tenure** is the #1 predictor — new customers churn far more often
- **Cashback amount** has a non-linear effect; very low cashback → high churn risk
- **Complaints** are a leading indicator — customers who complained churn at 3× the baseline
- **Days since last order** and **city tier** add strong signal
- SMOTE corrects for the 16.8% class imbalance, making models actionable in production

---

## ◈ Dataset Features

| Feature | Description |
|---|---|
| `tenure` | Months active |
| `satisfaction_score` | 1–5 rating |
| `complain` | Complaint raised (0/1) |
| `cashback_amount` | $ cashback received |
| `day_since_last_order` | Recency signal |
| `city_tier` | 1=Metro, 3=Small city |
| `churn` | **Target** — 1=churned |

---

## ◈ Quick Start

```bash
# 1. Clone
git clone https://github.com/isamkhan1809/customer-churn-prediction.git
cd customer-churn-prediction

# 2. Virtual environment
python -m venv venv && source venv/bin/activate

# 3. Install
pip install -r requirements.txt

# 4. Train the model (run notebook first)
jupyter notebook customer_churn_prediction.ipynb

# 5. Launch the app
python app.py
# → http://localhost:5000
```

---

## ◈ REST API

```bash
curl -X POST http://localhost:5000/api/predict \
     -H "Content-Type: application/json" \
     -d '{
       "tenure": 5,
       "satisfaction_score": 2,
       "complain": 1,
       "cashback_amount": 120,
       "day_since_last_order": 20,
       "city_tier": 3
     }'
```

---

## ◈ Tech Stack

| Layer | Technology |
|---|---|
| Modelling | scikit-learn, XGBoost |
| Imbalance | imbalanced-learn (SMOTE) |
| Deployment | Flask |
| Visualisation | Matplotlib, Seaborn |
| Notebook | Jupyter |

---

## ◈ Project Structure

```
customer-churn-prediction/
├── customer_churn_prediction.ipynb  ← Training pipeline
├── app.py                           ← Flask web app
├── templates/index.html             ← Prediction UI
├── requirements.txt
└── README.md
```

---

<div align="center">

**Predict. Retain. Grow.**

*MIT License*

</div>
