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

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

from xgboost import XGBClassifier

from ml.preprocessing import prepare_data


data = pd.read_csv(
    "dataset/Telco-Customer-Churn.csv"
)

X, y, preprocessor = prepare_data(data)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_train_transformed = preprocessor.fit_transform(
    X_train
)

X_test_transformed = preprocessor.transform(
    X_test
)


models = {

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    ),

    "XGBoost": XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )
}


results = []


for name, model in models.items():

    model.fit(
        X_train_transformed,
        y_train
    )

    predictions = model.predict(
        X_test_transformed
    )

    probabilities = model.predict_proba(
        X_test_transformed
    )[:, 1]

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(
            y_test,
            predictions
        ),
        "Precision": precision_score(
            y_test,
            predictions,
            zero_division=0
        ),
        "Recall": recall_score(
            y_test,
            predictions,
            zero_division=0
        ),
        "F1 Score": f1_score(
            y_test,
            predictions,
            zero_division=0
        ),
        "ROC-AUC": roc_auc_score(
            y_test,
            probabilities
        )
    })


ensemble_data = joblib.load(
    "models/ensemble_model.pkl"
)

ensemble = ensemble_data["model"]

ensemble_predictions = ensemble.predict(
    X_test_transformed
)

ensemble_probabilities = ensemble.predict_proba(
    X_test_transformed
)[:, 1]


results.append({
    "Model": "Soft-Voting Ensemble",
    "Accuracy": accuracy_score(
        y_test,
        ensemble_predictions
    ),
    "Precision": precision_score(
        y_test,
        ensemble_predictions,
        zero_division=0
    ),
    "Recall": recall_score(
        y_test,
        ensemble_predictions,
        zero_division=0
    ),
    "F1 Score": f1_score(
        y_test,
        ensemble_predictions,
        zero_division=0
    ),
    "ROC-AUC": roc_auc_score(
        y_test,
        ensemble_probabilities
    )
})


results_df = pd.DataFrame(results)

results_df = results_df.round(4)

print("\nModel Comparison\n")

print(
    results_df.to_string(
        index=False
    )
)


results_df.to_csv(
    "reports/results/all_model_comparison.csv",
    index=False
)

print(
    "\nResults saved to:",
    "reports/results/all_model_comparison.csv"
)