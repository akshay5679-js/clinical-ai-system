import shap
import joblib
import numpy as np
import os

model_path = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "ml",
    "heart_disease_model.pkl"
)

model = joblib.load(model_path)

def explain_prediction(input_data):

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(np.array(input_data))

    features = [
        "Age","Sex","Chest Pain","Blood Pressure","Cholesterol",
        "Fasting Blood Sugar","Rest ECG","Max Heart Rate",
        "Exercise Angina","Oldpeak","Slope","CA","Thal"
    ]

    contributions = dict(zip(features, shap_values[1][0]))

    sorted_contributions = sorted(
        contributions.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )

    return sorted_contributions[:5]