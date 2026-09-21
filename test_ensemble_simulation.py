import pandas as pd

from simulation.ensemble_simulator import (
    EnsembleSimulator
)

from simulation.advanced_scenario_generator import (
    AdvancedScenarioGenerator
)


data = pd.read_csv(
    "dataset/Telco-Customer-Churn.csv"
)

customer = data.iloc[[0]].copy()

simulator = EnsembleSimulator(
    "models/ensemble_model.pkl"
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

print("\nEnsemble Adaptive Simulation\n")

print(
    "Original Churn Probability:",
    round(
        original_probability * 100,
        2
    ),
    "%"
)

print(
    "Total Scenarios:",
    len(scenarios)
)

for index, scenario in enumerate(
    scenarios,
    start=1
):

    result = simulator.compare_states(
        customer,
        scenario
    )

    print(
        f"\nScenario {index}:",
        scenario
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