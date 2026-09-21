import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from ml.preprocessing import prepare_data


df = pd.read_csv("dataset/Telco-Customer-Churn.csv")

X, y, preprocessor = prepare_data(df)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

models = {
    "Random Forest": "models/random_forest_model.pkl",
    "Gradient Boosting": "models/gradient_boosting_model.pkl",
    "XGBoost": "models/xgboost_model.pkl"
}

results = []

for name, path in models.items():

    model = joblib.load(path)

    y_pred = model.predict(X_test)

    y_probability = model.predict_proba(X_test)[:, 1]

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_probability)
    })


results_df = pd.DataFrame(results)

print("\nMODEL COMPARISON")
print("----------------")

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

results_df.to_csv(
    "reports/results/model_comparison.csv",
    index=False
)

print("\nResults saved successfully!")