import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)
import pandas as pd

from simulation.simulator import CustomerSimulator
from simulation.advanced_scenario_generator import (
    AdvancedScenarioGenerator
)


data = pd.read_csv(
    "dataset/Telco-Customer-Churn.csv"
)

customer = data.iloc[[0]].copy()

simulator = CustomerSimulator(
    "models/xgboost_model.pkl"
)

generator = AdvancedScenarioGenerator(
    max_features_per_scenario=2,
    max_scenarios=20
)

scenarios = generator.generate_scenarios(
    customer
)

original_probability = simulator.predict_risk(
    customer
)

results = []

for scenario in scenarios:

    simulated_state = customer.copy()

    for feature, value in scenario.items():

        simulated_state.loc[
            simulated_state.index[0],
            feature
        ] = value

    simulated_probability = simulator.predict_risk(
        simulated_state
    )

    probability_change = (
        original_probability -
        simulated_probability
    )

    results.append({
        "scenario": scenario,
        "original_probability": original_probability,
        "simulated_probability": simulated_probability,
        "probability_change": probability_change
    })


results = sorted(
    results,
    key=lambda x: x["probability_change"],
    reverse=True
)


print("\nScenario Stability Analysis\n")

print(
    "Original Churn Probability:",
    round(original_probability * 100, 2),
    "%"
)

print(
    "Total Scenarios Tested:",
    len(results)
)

print("\nScenario Results:\n")

for index, result in enumerate(
    results,
    start=1
):

    output_data = []

for index, result in enumerate(
    results,
    start=1
):
    output_data.append({
        "scenario_number": index,
        "scenario": str(result["scenario"]),
        "original_probability": result["original_probability"],
        "simulated_probability": result["simulated_probability"],
        "probability_change": result["probability_change"]
    })

results_df = pd.DataFrame(output_data)

results_df.to_csv(
    "experiments/scenario_stability_results.csv",
    index=False
)

print(
    "\nResults saved to:",
    "experiments/scenario_stability_results.csv"
)

print(
        f"Scenario {index}:",
        result["scenario"]
    )

print(
        "Simulated Probability:",
        round(
            result["simulated_probability"] * 100,
            2
        ),
        "%"
    )

print(
        "Probability Change:",
        round(
            result["probability_change"] * 100,
            2
        ),
        "percentage points"
    )

print()