from itertools import combinations, product

from simulation.feasibility import (
    is_customer_state_valid,
    get_allowed_values
)


class AdvancedScenarioGenerator:

    def __init__(
        self,
        max_features_per_scenario=2,
        max_scenarios=20
    ):
        self.max_features_per_scenario = max_features_per_scenario
        self.max_scenarios = max_scenarios

    def generate_scenarios(self, customer_state):

        feature_changes = []

        for feature in customer_state.columns:

            allowed_values = get_allowed_values(feature)

            if not allowed_values:
                continue

            current_value = customer_state.iloc[0][feature]

            valid_values = []

            for value in allowed_values:

                if value == current_value:
                    continue

                if is_customer_state_valid(
                    customer_state,
                    feature,
                    value
                ):
                    valid_values.append(value)

            if valid_values:

                feature_changes.append(
                    (feature, valid_values)
                )

        scenarios = []

        for feature_count in range(
            1,
            self.max_features_per_scenario + 1
        ):

            for feature_group in combinations(
                feature_changes,
                feature_count
            ):

                features = [
                    item[0]
                    for item in feature_group
                ]

                value_lists = [
                    item[1]
                    for item in feature_group
                ]

                for values in product(*value_lists):

                    scenario = dict(
                        zip(features, values)
                    )

                    simulated_state = customer_state.copy()

                    for feature, value in scenario.items():

                        simulated_state.loc[
                            simulated_state.index[0],
                            feature
                        ] = value

                    valid = True

                    for feature, value in scenario.items():

                        if not is_customer_state_valid(
                            simulated_state,
                            feature,
                            value
                        ):
                            valid = False
                            break

                    if not valid:
                        continue

                    scenarios.append(scenario)

                    if len(scenarios) >= self.max_scenarios:
                        return scenarios

        return scenarios