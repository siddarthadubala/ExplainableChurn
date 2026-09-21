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

prediction = ensemble.predict(
    customer_transformed
)[0]

probability = ensemble.predict_proba(
    customer_transformed
)[0][1]

feature_names = (
    preprocessor
    .get_feature_names_out()
)

shap_values = shap_result.values[0]

explanation = pd.DataFrame({
    "feature": feature_names,
    "shap_value": shap_values
})

explanation["absolute_shap"] = (
    explanation["shap_value"].abs()
)

explanation = explanation.sort_values(
    "absolute_shap",
    ascending=False
)

print("\nCustomer Explanation\n")

print(
    "Predicted Churn:",
    "Yes" if prediction == 1 else "No"
)

print(
    "Predicted Churn Probability:",
    round(
        probability * 100,
        2
    ),
    "%"
)

print("\nTop Contributing Features:\n")

print(
    explanation[
        [
            "feature",
            "shap_value"
        ]
    ].head(10).to_string(
        index=False
    )
)