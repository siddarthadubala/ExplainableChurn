from simulation.feasibility import is_customer_state_valid
from simulation.feasibility import get_allowed_values


class ScenarioGenerator:

    def __init__(self, max_scenarios=10):
        self.max_scenarios = max_scenarios

    def generate_scenarios(self, customer_state):

        scenarios = []

        for feature in customer_state.columns:

            allowed_values = get_allowed_values(
                feature
            )

            if not allowed_values:
                continue

            current_value = customer_state.iloc[0][
                feature
            ]

            for value in allowed_values:

                if value == current_value:
                    continue

                if not is_customer_state_valid(
                    customer_state,
                    feature,
                    value
                ):
                    continue

                scenario = {
                    feature: value
                }

                scenarios.append(
                    scenario
                )

                if len(scenarios) >= self.max_scenarios:
                    return scenarios

        return scenarios