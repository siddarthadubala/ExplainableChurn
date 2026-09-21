import pandas as pd

from simulation.scenario_generator import ScenarioGenerator


data = pd.read_csv(
    "dataset/Telco-Customer-Churn.csv"
)

customer = data.iloc[[0]].copy()

generator = ScenarioGenerator(
    max_scenarios=10
)

scenarios = generator.generate_scenarios(
    customer
)

print("\nGenerated Scenarios:\n")

for i, scenario in enumerate(
    scenarios,
    start=1
):

    print(
        f"Scenario {i}: {scenario}"
    )