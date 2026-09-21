import pandas as pd
import joblib


class CustomerSimulator:

    def __init__(self, model_path):
        self.model = joblib.load(model_path)

    def predict_risk(self, customer_state):

        probability = self.model.predict_proba(
            customer_state
        )[0][1]

        return probability

    def simulate_change(
        self,
        customer_state,
        feature,
        new_value
    ):

        simulated_state = customer_state.copy()

        simulated_state.loc[
            simulated_state.index[0],
            feature
        ] = new_value

        probability = self.predict_risk(
            simulated_state
        )

        return simulated_state, probability

    def compare_states(
        self,
        original_state,
        feature,
        new_value
    ):

        original_probability = self.predict_risk(
            original_state
        )

        simulated_state, simulated_probability = (
            self.simulate_change(
                original_state,
                feature,
                new_value
            )
        )

        change = (
            original_probability -
            simulated_probability
        )

        return {
            "original_probability":
                original_probability,
            "simulated_probability":
                simulated_probability,
            "probability_change":
                change,
            "feature":
                feature,
            "new_value":
                new_value,
            "simulated_state":
                simulated_state
        }