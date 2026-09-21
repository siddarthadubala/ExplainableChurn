import pandas as pd
import joblib
import shap

from ml.preprocessing import prepare_data


df = pd.read_csv("dataset/Telco-Customer-Churn.csv")

X, y, preprocessor = prepare_data(df)

model = joblib.load("models/xgboost_model.pkl")

preprocessor_model = model.named_steps["preprocessor"]
classifier = model.named_steps["classifier"]

X_transformed = preprocessor_model.transform(X)

feature_names = preprocessor_model.get_feature_names_out()

explainer = shap.TreeExplainer(classifier)

shap_values = explainer.shap_values(X_transformed)

print("SHAP explanation generated successfully!")

print("Number of customers:", X.shape[0])

print("Number of processed features:", len(feature_names))

print("First 20 processed features:")

for feature in feature_names[:20]:
    print(feature)

print("SHAP values shape:", shap_values.shape)