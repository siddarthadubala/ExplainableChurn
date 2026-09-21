import pandas as pd

from simulation.advanced_scenario_generator import (
    AdvancedScenarioGenerator
)


data = pd.read_csv(
    "dataset/Telco-Customer-Churn.csv"
)

customer = data.iloc[[0]].copy()

generator = AdvancedScenarioGenerator(
    max_features_per_scenario=2,
    max_scenarios=20
)

scenarios = generator.generate_scenarios(
    customer
)

print("\nAdvanced Generated Scenarios:\n")

for index, scenario in enumerate(
    scenarios,
    start=1
):

    print(
        f"Scenario {index}: {scenario}"
    )

print(
    "\nTotal Scenarios:",
    len(scenarios)
)