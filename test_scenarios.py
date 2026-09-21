import pandas as pd

from simulation.state_engine import CustomerState
from simulation.simulator import CustomerSimulator
from simulation.scenario_manager import ScenarioManager


df = pd.read_csv(
    "dataset/Telco-Customer-Churn.csv"
)

customer = df.iloc[[0]].drop(
    columns=["Churn", "customerID"],
    errors="ignore"
)

state = CustomerState(customer)

simulator = CustomerSimulator(
    "models/xgboost_model.pkl"
)

manager = ScenarioManager(simulator)

original_state = state.get_state()

scenarios = [
    {
        "Contract": "One year"
    },
    {
        "Contract": "Two year"
    },
    {
        "TechSupport": "Yes"
    },
    {
        "Contract": "One year",
        "TechSupport": "Yes"
    }
]

results = manager.evaluate_multiple_scenarios(
    original_state,
    scenarios
)

print("\nCUSTOMER STATE SIMULATION")
print("-------------------------")

print(
    "Original churn probability:",
    round(
        simulator.predict_risk(original_state) * 100,
        2
    ),
    "%"
)

for number, result in enumerate(
    results,
    start=1
):

    print("\nScenario", number)

    print(
        "Changes:",
        result["changes"]
    )

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