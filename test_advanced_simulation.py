import pandas as pd

from simulation.simulator import CustomerSimulator
from simulation.scenario_manager import ScenarioManager
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

manager = ScenarioManager(
    simulator
)

generator = AdvancedScenarioGenerator(
    max_features_per_scenario=2,
    max_scenarios=20
)

scenarios = generator.generate_scenarios(
    customer
)

results = manager.evaluate_multiple_scenarios(
    customer,
    scenarios
)

results = manager.rank_scenarios(
    results
)

results = manager.add_impact_labels(
    results
)

results = manager.add_risk_direction(
    results
)

summary = manager.create_summary(
    results
)

print("\nAdvanced Simulation Results:\n")

for result in summary:

    print(
        f"\nScenario {result['scenario']}"
    )

    print(
        "Changes:",
        result["changes"]
    )

    print(
        "Original Churn Probability:",
        round(
            result["original_probability"] * 100,
            2
        ),
        "%"
    )

    print(
        "Simulated Churn Probability:",
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

    print(
        "Impact:",
        result["impact"]
    )

    print(
        "Risk Direction:",
        result["risk_direction"]
    )