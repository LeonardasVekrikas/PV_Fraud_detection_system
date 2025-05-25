import joblib
import numpy as np
import pandas as pd
from model_api.shap_lime import explain_with_shap_and_lime

_model = None
_feature_names = None

def get_lgbm_model():
    global _model
    if _model is None:
        _model = joblib.load("models/lightgbm_model.pkl")
    return _model

def get_feature_names():
    global _feature_names
    if _feature_names is None:
        _feature_names = joblib.load("models/features_lgbm.pkl")
    return _feature_names

def predict_lgbm(features):
    feature_names = get_feature_names()
    X_df = pd.DataFrame([features], columns=feature_names)
    model = get_lgbm_model()
    prob = model.predict_proba(X_df)[0][1]
    explanation = explain_with_shap_and_lime(model, X_df, feature_names)
    return prob, explanation
