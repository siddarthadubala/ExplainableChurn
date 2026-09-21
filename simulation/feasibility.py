ALLOWED_SCENARIO_FEATURES = {

    "Contract": [
        "Month-to-month",
        "One year",
        "Two year"
    ],

    "OnlineSecurity": [
        "Yes",
        "No",
        "No internet service"
    ],

    "OnlineBackup": [
        "Yes",
        "No",
        "No internet service"
    ],

    "DeviceProtection": [
        "Yes",
        "No",
        "No internet service"
    ],

    "TechSupport": [
        "Yes",
        "No",
        "No internet service"
    ],

    "StreamingTV": [
        "Yes",
        "No",
        "No internet service"
    ],

    "StreamingMovies": [
        "Yes",
        "No",
        "No internet service"
    ],

    "PaperlessBilling": [
        "Yes",
        "No"
    ]
}


def is_valid_scenario(
    feature,
    value
):

    if feature not in ALLOWED_SCENARIO_FEATURES:
        return False

    return value in ALLOWED_SCENARIO_FEATURES[feature]


def get_allowed_values(feature):

    return ALLOWED_SCENARIO_FEATURES.get(
        feature,
        []
    )
def is_customer_state_valid(
    customer_state,
    feature,
    value
):

    if not is_valid_scenario(
        feature,
        value
    ):
        return False

    if (
        feature in [
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies"
        ]
    ):

        internet_service = customer_state.iloc[0][
            "InternetService"
        ]

        if (
            internet_service == "No"
            and value != "No internet service"
        ):
            return False

        if (
            internet_service != "No"
            and value == "No internet service"
        ):
            return False

    return True