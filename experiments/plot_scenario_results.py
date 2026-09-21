import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv(
    "experiments/scenario_stability_results.csv"
)

data = data.sort_values(
    "probability_change",
    ascending=True
)

plt.figure(
    figsize=(10, 7)
)

plt.barh(
    range(len(data)),
    data["probability_change"] * 100
)

plt.yticks(
    range(len(data)),
    data["scenario"]
)

plt.xlabel(
    "Change in Predicted Churn Probability (percentage points)"
)

plt.ylabel(
    "Customer State Scenario"
)

plt.title(
    "Adaptive Customer-State Simulation Results"
)

plt.axvline(
    0,
    linewidth=1
)

plt.tight_layout()

plt.savefig(
    "experiments/scenario_simulation_results.png",
    dpi=300
)

plt.show()

print(
    "\nGraph saved to:",
    "experiments/scenario_simulation_results.png"
)