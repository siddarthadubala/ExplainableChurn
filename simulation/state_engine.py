import pandas as pd


class CustomerState:

    def __init__(self, customer_data):
        self.state = customer_data.copy()

    def get_state(self):
        return self.state.copy()

    def update(self, feature, value):
        if feature not in self.state.columns:
            raise ValueError(
                f"Feature '{feature}' not found."
            )

        self.state.loc[self.state.index[0], feature] = value

    def reset(self, original_state):
        self.state = original_state.copy()