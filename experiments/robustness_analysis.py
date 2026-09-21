import os
import sys
import joblib
import pandas as pd

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from ml.preprocessing import prepare_data
from simulation.advanced_scenario_generator import AdvancedScenarioGenerator
from simulation.ensemble_simulator import EnsembleSimulator

DATASET_PATH = "dataset/Telco-Customer-Churn.csv"
MODEL_PATH = "models/ensemble_model.pkl"
OUTPUT_PATH = "experiments/robustness_results.csv"

data = pd.read_csv(DATASET_PATH)

if "TotalCharges" in data.columns:
    data["TotalCharges"] = pd.to_numeric(
        data["TotalCharges"],
        errors="coerce"
    )

X, y, preprocessor = prepare_data(data)

model_data = joblib.load(MODEL_PATH)
model = model_data["model"]

preprocessor.fit(X)

X_transformed = preprocessor.transform(X)

probabilities = model.predict_proba(X_transformed)[:, 1]

data["churn_probability"] = probabilities

high_risk = data[
    data["churn_probability"] >= 0.50
].copy()

high_risk = high_risk.sort_values(
    "churn_probability",
    ascending=False
).head(50)

generator = AdvancedScenarioGenerator(
    max_features_per_scenario=2,
    max_scenarios=20
)

simulator = EnsembleSimulator(MODEL_PATH)

results = []

for _, customer in high_risk.iterrows():

    customer_id = customer["customerID"]

    customer_state = pd.DataFrame([
        customer.drop(
            labels=["churn_probability"]
        )
    ])

    scenarios = generator.generate_scenarios(
        customer_state
    )

    original_probability = simulator.predict_risk(
        customer_state
    )

    for scenario in scenarios:

        simulated_state, simulated_probability = (
            simulator.simulate_change(
                customer_state,
                scenario
            )
        )

        probability_change = (
            original_probability -
            simulated_probability
        )

        results.append({
            "customerID": customer_id,
            "original_probability": original_probability,
            "simulated_probability": simulated_probability,
            "probability_change": probability_change
        })

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_PATH,
    index=False
)

risk_decreased = (
    results_df["probability_change"] > 0
).sum()

risk_increased = (
    results_df["probability_change"] < 0
).sum()

print("\nROBUSTNESS ANALYSIS")
print("--------------------")

print(
    "Customers tested:",
    high_risk.shape[0]
)

print(
    "Total scenarios:",
    len(results_df)
)

print(
    "Average original probability:",
    round(
        results_df[
            "original_probability"
        ].mean(),
        4
    )
)

print(
    "Average simulated probability:",
    round(
        results_df[
            "simulated_probability"
        ].mean(),
        4
    )
)

print(
    "Average probability change:",
    round(
        results_df[
            "probability_change"
        ].mean(),
        4
    )
)

print(
    "Median probability change:",
    round(
        results_df[
            "probability_change"
        ].median(),
        4
    )
)

print(
    "Standard deviation:",
    round(
        results_df[
            "probability_change"
        ].std(),
        4
    )
)

print(
    "Risk-decreasing scenarios:",
    risk_decreased
)

print(
    "Risk-increasing scenarios:",
    risk_increased
)

print("\nSaved to:")
print(OUTPUT_PATH)