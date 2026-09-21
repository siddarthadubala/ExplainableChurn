from flask import Flask, render_template, request, session
import pandas as pd
import joblib
import shap
import os

from simulation.advanced_scenario_generator import AdvancedScenarioGenerator
from simulation.scenario_manager import ScenarioManager
from simulation.ensemble_simulator import EnsembleSimulator


app = Flask(__name__)

app.secret_key = "churnsense-ai-secret-key"

UPLOAD_FOLDER = "uploaded_data"

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():

    file = request.files.get("file")

    if file is None or file.filename == "":
        return "No file selected."

    if not file.filename.lower().endswith(".csv"):
        return "Please upload a CSV file."

    df = pd.read_csv(file)

    uploaded_path = os.path.join(
        UPLOAD_FOLDER,
        "current_dataset.csv"
    )

    df.to_csv(
        uploaded_path,
        index=False
    )

    session["uploaded_dataset"] = uploaded_path

    df = df.replace(
        r"^\s*$",
        pd.NA,
        regex=True
    )

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"],
            errors="coerce"
        )

    missing = int(
        df.isnull().sum().sum()
    )

    missing_by_column = (
        df.isnull().sum().to_dict()
    )

    if "Churn" in df.columns:
        churn_counts = (
            df["Churn"].value_counts().to_dict()
        )
    else:
        churn_counts = {}

    model_data = joblib.load(
        "models/ensemble_model.pkl"
    )

    model = model_data["model"]

    preprocessor = model_data["preprocessor"]

    prediction_data = df.drop(
        columns=[
            "Churn",
            "customerID"
        ],
        errors="ignore"
    )

    transformed_data = preprocessor.transform(
        prediction_data
    )

    probabilities = model.predict_proba(
        transformed_data
    )[:, 1]

    results = df.copy()

    results["Churn Probability"] = (
        probabilities * 100
    ).round(2)

    results["Risk"] = results[
        "Churn Probability"
    ].apply(
        lambda x:
        "High"
        if x >= 70
        else "Medium"
        if x >= 40
        else "Low"
    )

    results["Prediction"] = results[
        "Churn Probability"
    ].apply(
        lambda x:
        "Likely to Churn"
        if x >= 70
        else "Likely to Stay"
    )

    high_risk = results[
        results["Risk"] == "High"
    ].sort_values(
        by="Churn Probability",
        ascending=False
    )

    prediction_records = high_risk.to_dict(
        "records"
    )

    return render_template(
        "analysis.html",
        rows=len(df),
        columns=len(df.columns),
        missing=missing,
        column_names=df.columns.tolist(),
        missing_by_column=missing_by_column,
        churn_counts=churn_counts,
        prediction_records=prediction_records,
        total_high_risk=len(high_risk)
    )


@app.route("/simulation", methods=["GET", "POST"])
def simulation():

    customer = None
    customer_id = None
    probability = None
    explanations = []
    scenario_results = []

    risk_decreased = 0
    risk_increased = 0
    total_scenarios = 0

    chart_data = []

    best_scenario = None

    if request.method == "POST":

        customer_id = request.form.get(
            "customer_id"
        )

        uploaded_path = session.get(
            "uploaded_dataset"
        )

        if uploaded_path and os.path.exists(
            uploaded_path
        ):
            dataset_path = uploaded_path
        else:
            dataset_path = (
                "dataset/Telco-Customer-Churn.csv"
            )

        data = pd.read_csv(
            dataset_path
        )

        data = data.replace(
            r"^\s*$",
            pd.NA,
            regex=True
        )

        if "TotalCharges" in data.columns:
            data["TotalCharges"] = pd.to_numeric(
                data["TotalCharges"],
                errors="coerce"
            )

        if "customerID" not in data.columns:
            return render_template(
                "simulation.html",
                customer=None,
                customer_id=customer_id,
                probability=None,
                explanations=[],
                scenario_results=[],
                risk_decreased=0,
                risk_increased=0,
                total_scenarios=0,
                chart_data=[],
                best_scenario=None
            )

        customer = data[
            data["customerID"].astype(str)
            == str(customer_id)
        ]

        if not customer.empty:

            customer_features = customer.drop(
                columns=[
                    "Churn",
                    "customerID"
                ],
                errors="ignore"
            )

            model_data = joblib.load(
                "models/ensemble_model.pkl"
            )

            model = model_data["model"]

            preprocessor = (
                model_data["preprocessor"]
            )

            transformed_data = (
                preprocessor.transform(
                    customer_features
                )
            )

            probability = (
                model.predict_proba(
                    transformed_data
                )[0][1]
            )

            probability = round(
                probability * 100,
                2
            )

            xgb_model = (
                model.named_estimators_[
                    "xgboost"
                ]
            )

            explainer = shap.TreeExplainer(
                xgb_model
            )

            shap_values = (
                explainer.shap_values(
                    transformed_data
                )
            )

            if isinstance(
                shap_values,
                list
            ):
                shap_values = shap_values[1]

            shap_values = shap_values[0]

            feature_names = (
                preprocessor
                .get_feature_names_out()
            )

            explanation_data = pd.DataFrame(
                {
                    "feature": feature_names,
                    "shap_value": shap_values
                }
            )

            explanation_data[
                "absolute_shap"
            ] = explanation_data[
                "shap_value"
            ].abs()

            explanation_data = (
                explanation_data
                .sort_values(
                    "absolute_shap",
                    ascending=False
                )
                .head(10)
            )

            for _, row in (
                explanation_data.iterrows()
            ):

                feature = row["feature"]

                feature = feature.replace(
                    "num__",
                    ""
                ).replace(
                    "cat__",
                    ""
                )

                value = round(
                    float(
                        row["shap_value"]
                    ),
                    4
                )

                if value > 0:
                    direction = (
                        "Increases churn risk"
                    )
                else:
                    direction = (
                        "Decreases churn risk"
                    )

                explanations.append(
                    {
                        "feature": feature,
                        "shap_value": value,
                        "direction": direction
                    }
                )

            simulator = EnsembleSimulator(
                "models/ensemble_model.pkl"
            )

            generator = (
                AdvancedScenarioGenerator(
                    max_features_per_scenario=2,
                    max_scenarios=20
                )
            )

            manager = ScenarioManager(
                simulator
            )

            scenarios = (
                generator.generate_scenarios(
                    customer_features
                )
            )

            results = (
                manager.evaluate_multiple_scenarios(
                    customer_features,
                    scenarios
                )
            )

            results = manager.rank_scenarios(
                results
            )

            results = (
                manager.add_impact_labels(
                    results
                )
            )

            results = (
                manager.add_risk_direction(
                    results
                )
            )

            scenario_results = (
                manager.create_summary(
                    results
                )
            )

            chart_data = []

            for result in scenario_results:

                chart_data.append(
                    {
                        "scenario":
                            result["scenario"],

                        "original":
                            round(
                                result[
                                    "original_probability"
                                ] * 100,
                                2
                            ),

                        "simulated":
                            round(
                                result[
                                    "simulated_probability"
                                ] * 100,
                                2
                            )
                    }
                )

            risk_decreased = sum(
                1
                for result in scenario_results
                if result[
                    "probability_change"
                ] > 0
            )

            risk_increased = sum(
                1
                for result in scenario_results
                if result[
                    "probability_change"
                ] < 0
            )

            total_scenarios = len(
                scenario_results
            )

            decreasing_scenarios = [
                result
                for result in scenario_results
                if result[
                    "probability_change"
                ] > 0
            ]

            if decreasing_scenarios:

                best_scenario = max(
                    decreasing_scenarios,
                    key=lambda x:
                    x["probability_change"]
                )

            customer = customer.to_dict(
                "records"
            )[0]

        else:

            customer = None

    return render_template(
        "simulation.html",
        customer=customer,
        customer_id=customer_id,
        probability=probability,
        explanations=explanations,
        scenario_results=scenario_results,
        risk_decreased=risk_decreased,
        risk_increased=risk_increased,
        total_scenarios=total_scenarios,
        chart_data=chart_data,
        best_scenario=best_scenario
    )


if __name__ == "__main__":
    app.run(debug=True)