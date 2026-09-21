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

customer = X_test.iloc[[0]]

customer_transformed = preprocessor.transform(
    customer
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

shap_result = explainer(
    customer_transformed
)

shap_values = shap_result.values[0]

feature_names = (
    preprocessor.get_feature_names_out()
)

explanation = pd.DataFrame({
    "feature": feature_names,
    "shap_value": shap_values
})

explanation["absolute_shap"] = (
    explanation["shap_value"].abs()
)

positive = explanation[
    explanation["shap_value"] > 0
].sort_values(
    "shap_value",
    ascending=False
)

negative = explanation[
    explanation["shap_value"] < 0
].sort_values(
    "shap_value"
)

print("\nHuman-Readable Customer Explanation\n")

print("Factors Increasing Predicted Churn Risk:\n")

for _, row in positive.head(5).iterrows():

    feature = row["feature"]
    value = row["shap_value"]

    feature = feature.replace(
        "num__",
        ""
    )

    feature = feature.replace(
        "cat__",
        ""
    )

    print(
        f"{feature}: "
        f"+{value:.4f}"
    )

print("\nFactors Decreasing Predicted Churn Risk:\n")

for _, row in negative.head(5).iterrows():

    feature = row["feature"]
    value = row["shap_value"]

    feature = feature.replace(
        "num__",
        ""
    )

    feature = feature.replace(
        "cat__",
        ""
    )

    print(
        f"{feature}: "
        f"{value:.4f}"
    )