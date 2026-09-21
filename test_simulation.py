import pandas as pd

from simulation.state_engine import CustomerState
from simulation.simulator import CustomerSimulator


df = pd.read_csv(
    "dataset/Telco-Customer-Churn.csv"
)

customer = df.iloc[[0]].drop(
    columns=["Churn"],
    errors="ignore"
)

customer = customer.drop(
    columns=["customerID"],
    errors="ignore"
)


state = CustomerState(customer)

simulator = CustomerSimulator(
    "models/xgboost_model.pkl"
)


original_state = state.get_state()

original_probability = simulator.predict_risk(
    original_state
)

print("Original churn probability:")
print(round(original_probability * 100, 2), "%")


if "Contract" in original_state.columns:

    contract_values = [
        "One year",
        "Two year"
    ]

    for value in contract_values:

        result = simulator.compare_states(
            original_state,
            "Contract",
            value
        )

        print("\nSimulation:")
        print("Contract =", value)

        print(
            "Simulated churn probability:",
            round(
                result["simulated_probability"] * 100,
                2
            ),
            "%"
        )

        print(
            "Probability change:",
            round(
                result["probability_change"] * 100,
                2
            ),
            "percentage points"
        )