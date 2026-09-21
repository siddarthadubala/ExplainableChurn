from simulation.feasibility import is_valid_scenario


class ScenarioManager:

    def __init__(self, simulator):
        self.simulator = simulator

    def evaluate_scenario(
        self,
        original_state,
        changes
    ):

        simulated_state = original_state.copy()

        for feature, value in changes.items():

            if feature not in simulated_state.columns:
                raise ValueError(
                    f"Feature '{feature}' not found."
                )

            if not is_valid_scenario(
                feature,
                value
            ):
                raise ValueError(
                    f"Invalid scenario: "
                    f"{feature} = {value}"
                )

            simulated_state.loc[
                simulated_state.index[0],
                feature
            ] = value

        original_probability = (
            self.simulator.predict_risk(
                original_state
            )
        )

        simulated_probability = (
            self.simulator.predict_risk(
                simulated_state
            )
        )

        probability_change = (
            original_probability -
            simulated_probability
        )

        return {
            "original_state":
                original_state.copy(),

            "changes":
                changes,

            "simulated_state":
                simulated_state.copy(),

            "original_probability":
                original_probability,

            "simulated_probability":
                simulated_probability,

            "probability_change":
                probability_change
        }

    def evaluate_multiple_scenarios(
        self,
        original_state,
        scenarios
    ):

        results = []

        for scenario in scenarios:

            result = self.evaluate_scenario(
                original_state,
                scenario
            )

            results.append(result)

        return results

    def rank_scenarios(
        self,
        results
    ):

        ranked_results = sorted(
            results,
            key=lambda x: x["probability_change"],
            reverse=True
        )

        return ranked_results

    def add_impact_labels(
        self,
        results
    ):

        for result in results:

            change = result["probability_change"]

            if change >= 0.20:
                result["impact"] = "High"

            elif change >= 0.10:
                result["impact"] = "Medium"

            else:
                result["impact"] = "Low"

        return results

    def add_risk_direction(
        self,
        results
    ):

        for result in results:

            change = result["probability_change"]

            if change > 0:
                result["risk_direction"] = "Decreased"

            elif change < 0:
                result["risk_direction"] = "Increased"

            else:
                result["risk_direction"] = "Unchanged"

        return results

    def create_summary(
        self,
        results
    ):

        summary = []

        for index, result in enumerate(
            results,
            start=1
        ):

            summary.append({
                "scenario":
                    index,

                "changes":
                    result["changes"],

                "original_probability":
                    result["original_probability"],

                "simulated_probability":
                    result["simulated_probability"],

                "probability_change":
                    result["probability_change"],

                "impact":
                    result["impact"],

                "risk_direction":
                    result["risk_direction"]
            })

        return summary