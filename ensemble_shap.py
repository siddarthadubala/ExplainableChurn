import pandas as pd
import joblib
import shap

from sklearn.model_selection import train_test_split

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

ensemble_data = joblib.load(
    "models/ensemble_model.pkl"
)

ensemble = ensemble_data["model"]

xgboost_model = ensemble.named_estimators_[
    "xgboost"
]

explainer = shap.TreeExplainer(
    xgboost_model
)

shap_values = explainer(
    X_test_transformed
)

feature_names = (
    preprocessor
    .get_feature_names_out()
)

print("\nEnsemble SHAP Explanation\n")

print(
    "Test samples:",
    X_test.shape[0]
)

print(
    "Transformed features:",
    X_test_transformed.shape[1]
)

print(
    "SHAP values generated successfully."
)

print(
    "Feature names:",
    len(feature_names)
)

print(
    "SHAP matrix shape:",
    shap_values.values.shape
)