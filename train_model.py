import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

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
        RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced"
        )
    )
])

model.fit(X_train, y_train)

joblib.dump(model, "models/random_forest_model.pkl")

print("Model training completed successfully!")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Model saved successfully!")
