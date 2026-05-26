import numpy as np
from flask import Flask, request, render_template, jsonify
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

app = Flask(__name__)

# ── Global state ──────────────────────────────────────────────────────────────
model = None
scaler = None
feature_names = []
feature_importances = []
model_ready = False
model_accuracy = 0.0
model_f1 = 0.0
dataset_churn_rate = 0.0
dataset_avg_tenure = 0.0

FEATURE_NAMES = [
    "tenure",
    "city_tier",
    "warehouse_to_home",
    "hour_spend_on_app",
    "num_device_registered",
    "satisfaction_score",
    "num_address",
    "complain",
    "order_amount_hike_from_last_year",
    "coupon_used",
    "order_count",
    "day_since_last_order",
    "cashback_amount",
    "gender",
    "marital_status",
]

N_SAMPLES = 5000


def generate_dataset(seed: int = 42) -> tuple:
    """Generate a synthetic e-commerce churn dataset with ~16.8% positive rate."""
    rng = np.random.default_rng(seed)

    tenure = rng.integers(0, 61, N_SAMPLES).astype(float)
    city_tier = rng.choice([1, 2, 3], N_SAMPLES, p=[0.3, 0.4, 0.3]).astype(float)
    warehouse_to_home = rng.integers(5, 131, N_SAMPLES).astype(float)
    hour_spend_on_app = np.clip(rng.normal(3.0, 1.0, N_SAMPLES), 0, 5)
    num_device_registered = rng.integers(1, 11, N_SAMPLES).astype(float)
    satisfaction_score = rng.integers(1, 6, N_SAMPLES).astype(float)
    num_address = rng.integers(1, 21, N_SAMPLES).astype(float)
    complain = rng.choice([0, 1], N_SAMPLES, p=[0.85, 0.15]).astype(float)
    order_amount_hike = np.clip(rng.normal(15.0, 3.0, N_SAMPLES), 11, 26)
    coupon_used = rng.integers(0, 17, N_SAMPLES).astype(float)
    order_count = rng.integers(0, 26, N_SAMPLES).astype(float)
    day_since_last_order = rng.integers(0, 46, N_SAMPLES).astype(float)
    cashback_amount = np.clip(rng.normal(170.0, 60.0, N_SAMPLES), 0, 500)
    gender = rng.choice([0, 1], N_SAMPLES).astype(float)
    marital_status = rng.choice([0, 1, 2], N_SAMPLES, p=[0.4, 0.45, 0.15]).astype(float)

    X = np.column_stack([
        tenure, city_tier, warehouse_to_home, hour_spend_on_app,
        num_device_registered, satisfaction_score, num_address, complain,
        order_amount_hike, coupon_used, order_count, day_since_last_order,
        cashback_amount, gender, marital_status,
    ])

    # Logistic function to produce realistic churn probabilities (~16.8% rate)
    log_odds = (
        -3.5
        - 0.06 * tenure
        + 0.3 * city_tier
        + 0.01 * warehouse_to_home
        - 0.2 * hour_spend_on_app
        + 0.05 * num_device_registered
        - 0.4 * satisfaction_score
        + 0.05 * num_address
        + 1.2 * complain
        - 0.02 * order_amount_hike
        - 0.05 * coupon_used
        - 0.04 * order_count
        + 0.04 * day_since_last_order
        - 0.004 * cashback_amount
        + 0.1 * gender
        + 0.1 * marital_status
        + rng.normal(0, 0.5, N_SAMPLES)
    )
    prob = 1.0 / (1.0 + np.exp(-log_odds))
    y = (prob > 0.5).astype(int)

    return X, y


def train_model() -> None:
    """Generate data, preprocess, train XGBoost, and populate globals."""
    global model, scaler, feature_names, feature_importances
    global model_ready, model_accuracy, model_f1
    global dataset_churn_rate, dataset_avg_tenure

    X, y = generate_dataset()
    dataset_churn_rate = float(y.mean())
    dataset_avg_tenure = float(X[:, 0].mean())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    clf = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
    )
    clf.fit(X_train_sc, y_train)

    y_pred = clf.predict(X_test_sc)
    model_accuracy = float(accuracy_score(y_test, y_pred))
    model_f1 = float(f1_score(y_test, y_pred))

    feature_names = FEATURE_NAMES[:]
    raw_imp = clf.feature_importances_
    feature_importances = [
        {"feature": name, "importance": float(imp)}
        for name, imp in sorted(
            zip(feature_names, raw_imp), key=lambda x: x[1], reverse=True
        )
    ]

    model = clf
    model_ready = True
    print(f"[INFO] Model trained successfully. Accuracy: {model_accuracy * 100:.1f}%")


# ── Train on startup ──────────────────────────────────────────────────────────
train_model()


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/api/status", methods=["GET"])
def status():
    return jsonify({
        "model_ready": model_ready,
        "accuracy": round(model_accuracy, 4),
        "f1": round(model_f1, 4),
    })


@app.route("/api/predict", methods=["POST"])
def api_predict():
    if not model_ready:
        return jsonify({"error": "Model not ready"}), 503

    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "No JSON payload received"}), 400

    try:
        row = np.array(
            [float(data.get(f, 0)) for f in FEATURE_NAMES], dtype=np.float64
        ).reshape(1, -1)

        row_scaled = scaler.transform(row)
        prediction = int(model.predict(row_scaled)[0])
        churn_prob = float(model.predict_proba(row_scaled)[0][1])

        # Compute per-feature contribution: importance * |normalised feature value|
        raw_imp = model.feature_importances_
        row_norm = np.abs(row_scaled[0])  # already zero-mean unit-variance
        scores = raw_imp * row_norm
        top3_idx = np.argsort(scores)[::-1][:3]
        risk_factors = [FEATURE_NAMES[i] for i in top3_idx]

        return jsonify({
            "prediction": prediction,
            "churn_probability": round(churn_prob, 4),
            "label": "Churn Risk" if prediction == 1 else "Likely to Stay",
            "risk_factors": risk_factors,
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/feature-importance", methods=["GET"])
def api_feature_importance():
    return jsonify(feature_importances)


@app.route("/api/dashboard-stats", methods=["GET"])
def api_dashboard_stats():
    # The top churn reason is the most important feature overall
    top_reason = feature_importances[0]["feature"].replace("_", " ").title() if feature_importances else "N/A"
    return jsonify({
        "total_customers": N_SAMPLES,
        "churn_rate": round(dataset_churn_rate * 100, 2),
        "avg_tenure": round(dataset_avg_tenure, 1),
        "top_churn_reason": top_reason,
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
