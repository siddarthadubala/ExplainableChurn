import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier
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

random_forest = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42
)

gradient_boosting = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

xgboost_model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)

ensemble = VotingClassifier(
    estimators=[
        ("random_forest", random_forest),
        ("gradient_boosting", gradient_boosting),
        ("xgboost", xgboost_model)
    ],
    voting="soft"
)

ensemble.fit(
    X_train_transformed,
    y_train
)

joblib.dump(
    {
        "model": ensemble,
        "preprocessor": preprocessor
    },
    "models/ensemble_model.pkl"
)

print("\nEnsemble model trained successfully.")

print(
    "Training samples:",
    X_train.shape[0]
)

print(
    "Testing samples:",
    X_test.shape[0]
)

print(
    "Model saved to:",
    "models/ensemble_model.pkl"
)