<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,12,17&height=200&section=header&text=Churn%20Predict&fontSize=80&fontColor=fff&animation=twinkling&fontAlignY=35&desc=Know%20Who%27s%20Leaving%20%E2%80%94%20Before%20They%20Leave&descAlignY=60&descSize=20" width="100%"/>

<br/>

[![Python](https://img.shields.io/badge/Python-3.9%2B-FF4B4B?style=for-the-badge&logo=python&logoColor=white&labelColor=0D0D0D)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-93%25%20Accuracy-FF6B35?style=for-the-badge&logoColor=white&labelColor=0D0D0D)](https://xgboost.readthedocs.io)
[![Flask](https://img.shields.io/badge/Flask-Deployed-000000?style=for-the-badge&logo=flask&logoColor=white&labelColor=0D0D0D)](https://flask.palletsprojects.com)
[![License](https://img.shields.io/badge/License-MIT-FF4B4B?style=for-the-badge&labelColor=0D0D0D)](LICENSE)

<br/>

<a href="https://github.com/isamkhan1809/customer-churn-prediction">
  <img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=700&size=22&pause=1000&color=FF4B4B&center=true&vCenter=true&width=700&lines=XGBoost+Churn+Classifier+%E2%80%94+93%25+Accuracy;SMOTE+%7C+Random+Forest+%7C+Logistic+Regression;Predict+Churn+Before+It+Happens;Every+Lost+Customer+Is+a+Revenue+Leak." alt="Typing SVG" />
</a>

</div>

---

<br/>

<div align="center">

```
  ╔══════════════════════════════════════════════════════════════╗
  ║                                                              ║
  ║   Every month, customers quietly decide to leave.           ║
  ║   No warning. No goodbye. Just a gap in the revenue.        ║
  ║                                                              ║
  ║       This model sees it coming three steps ahead.          ║
  ║                                                              ║
  ╚══════════════════════════════════════════════════════════════╝
```

</div>

<br/>

## `>_ The Story`

> *A customer complains once and never orders again. Another quietly drifts — fewer visits, smaller baskets, longer gaps between orders. By the time the churn shows in the data, it's already happened.*
>
> *This project builds a machine learning pipeline that reads the early signals — tenure, cashback patterns, complaint history — and flags the customers most likely to leave before they do.*
>
> *93% accuracy. Deployed as a live Flask app with a REST API.*

<br/>

## `>_ Predictions`

<table>
<tr>
<td width="50%">

**Customer signals in:**
```
tenure:              5 months
satisfaction_score:  2 / 5
complain:            1
cashback_amount:     $120
day_since_last_order: 20
city_tier:           3
```

</td>
<td width="50%">

**Churn risk out:**
```
🔴 HIGH RISK — 87% churn probability

Top signals:
  ↑ Complaint raised
  ↑ Low tenure (< 6 months)
  ↑ Low cashback amount
  ↑ 20 days since last order
```

</td>
</tr>
</table>

<br/>

## `>_ The Pipeline`

```
┌─────────────────────────────────────────────────────────────┐
│                   CHURN PREDICTION PIPELINE                 │
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │  5,000 rows  │───▶│  Feature Eng │───▶│    SMOTE     │  │
│  │  16 features │    │  Scaling     │    │  Balancing   │  │
│  │  16.8% churn │    │  Encoding    │    │  +12% recall │  │
│  └──────────────┘    └──────────────┘    └──────┬───────┘  │
│                                                 │          │
│                           ┌─────────────────────▼───────┐  │
│                           │  Logistic Regression  ~84%  │  │
│                           │  Random Forest        ~91%  │  │
│                           │  XGBoost ✦            ~93%  │  │
│                           └─────────────────────┬───────┘  │
│                                                 │          │
│                           ┌─────────────────────▼───────┐  │
│                           │     Flask Web App           │  │
│                           │     REST API /predict       │  │
│                           └─────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

<br/>

## `>_ Model Performance`

<div align="center">

| Model | Accuracy | F1 (Churn) | ROC-AUC |
|---|---|---|---|
| Logistic Regression | ~84% | ~0.60 | ~0.87 |
| Random Forest | ~91% | ~0.75 | ~0.96 |
| **XGBoost** ✦ | **~93%** | **~0.80** | **~0.97** |

</div>

<br/>

## `>_ Get Running`

```bash
# Clone
git clone https://github.com/isamkhan1809/customer-churn-prediction.git
cd customer-churn-prediction

# Install
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Train (run notebook first)
jupyter notebook customer_churn_prediction.ipynb

# Launch app
python app.py
```

Open [http://localhost:5000](http://localhost:5000) — fill in customer details, get a churn prediction.

<br/>

## `>_ REST API`

```bash
curl -X POST http://localhost:5000/api/predict \
     -H "Content-Type: application/json" \
     -d '{"tenure": 5, "satisfaction_score": 2, "complain": 1,
          "cashback_amount": 120, "day_since_last_order": 20, "city_tier": 3}'
```

<br/>

## `>_ Tech Stack`

<div align="center">

| Layer | Technology |
|---|---|
| **Modelling** | scikit-learn, XGBoost |
| **Imbalance** | imbalanced-learn (SMOTE) |
| **Deployment** | Flask |
| **Visualisation** | Matplotlib, Seaborn |
| **Notebook** | Jupyter |

</div>

<br/>

## `>_ Project Structure`

```
customer-churn-prediction/
├── customer_churn_prediction.ipynb  ← Full training pipeline
├── app.py                           ← Flask web app + REST API
├── templates/index.html             ← Prediction UI
└── requirements.txt
```

<br/>

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,12,17&height=120&section=footer&animation=twinkling" width="100%"/>

<br/>

*Predict. Retain. Grow.*
*93% accuracy. Built with Python, XGBoost, and Flask.*

<br/>

[![GitHub](https://img.shields.io/badge/github-isamkhan1809-FF4B4B?style=for-the-badge&logo=github&logoColor=white&labelColor=0D0D0D)](https://github.com/isamkhan1809)

</div>
