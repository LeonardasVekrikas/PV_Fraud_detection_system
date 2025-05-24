import joblib
import numpy as np
import pandas as pd
from model_api.shap_lime import explain_with_shap_and_lime

model = joblib.load("models/lightgbm_model.pkl")
feature_names = joblib.load("models/features_lgbm.pkl")

def predict_lgbm(features):
    X_df = pd.DataFrame([features], columns=feature_names)
    prob = model.predict_proba(X_df)[0][1]
    explanation = explain_with_shap_and_lime(model, X_df, feature_names)
    return prob, explanation
