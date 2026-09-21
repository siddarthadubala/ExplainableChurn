import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

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

model = Pipeline([
    ("preprocessor", preprocessor),
    (
        "classifier",
        XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42
        )
    )
])

model.fit(X_train, y_train)

joblib.dump(
    model,
    "models/xgboost_model.pkl"
)

print("XGBoost training completed!")
print("Model saved successfully!")