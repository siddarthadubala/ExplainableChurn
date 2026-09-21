import joblib


class EnsembleSimulator:

    def __init__(self, model_path):

        model_data = joblib.load(
            model_path
        )

        self.model = model_data["model"]
        self.preprocessor = model_data["preprocessor"]

    def predict_risk(self, customer_state):

        transformed_state = self.preprocessor.transform(
            customer_state
        )

        probability = self.model.predict_proba(
            transformed_state
        )[0][1]

        return probability

    def simulate_change(
        self,
        customer_state,
        changes
    ):

        simulated_state = customer_state.copy()

        for feature, value in changes.items():

            simulated_state.loc[
                simulated_state.index[0],
                feature
            ] = value

        probability = self.predict_risk(
            simulated_state
        )

        return (
            simulated_state,
            probability
        )

    def compare_states(
        self,
        original_state,
        changes
    ):

        original_probability = self.predict_risk(
            original_state
        )

        simulated_state, simulated_probability = (
            self.simulate_change(
                original_state,
                changes
            )
        )

        probability_change = (
            original_probability -
            simulated_probability
        )

        return {
            "original_probability":
                original_probability,

            "simulated_probability":
                simulated_probability,

            "probability_change":
                probability_change,

            "changes":
                changes,

            "simulated_state":
                simulated_state
        }