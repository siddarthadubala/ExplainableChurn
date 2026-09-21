import pandas as pd
from ml.preprocessing import prepare_data


df = pd.read_csv("dataset/Telco-Customer-Churn.csv")

X, y, preprocessor = prepare_data(df)

print("Dataset loaded successfully")
print("Original rows:", len(df))
print("Original columns:", len(df.columns))

print("Features:", X.shape)
print("Target:", y.shape)

print("Target values:")
print(y.value_counts())

print("Numerical columns:")
print(X.select_dtypes(exclude=["object"]).columns.tolist())

print("Categorical columns:")
print(X.select_dtypes(include=["object"]).columns.tolist())