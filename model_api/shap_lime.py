import shap
import lime.lime_tabular
import numpy as np

def explain_with_shap_and_lime(model, X, feature_names=None):
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(X.shape[1])]

    # --- SHAP
    shap_explainer = shap.TreeExplainer(model)
    shap_values = shap_explainer.shap_values(X)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # for binary classifiers
    shap_values_row = shap_values[0][:len(feature_names)]

    # --- LIME: Use full input distribution (if available) or fallback
    try:
        training_data = X.values
        if training_data.shape[0] < 10:
            training_data = np.tile(X.values[0], (50, 1))

        lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data,
            feature_names=feature_names,
            mode="classification",
            discretize_continuous=True,
            verbose=False
        )

        lime_explanation = lime_explainer.explain_instance(
            X.values[0], model.predict_proba, num_features=10
        )
        lime_results = lime_explanation.as_list()
    except Exception as e:
        print("[LIME ERROR]", e)
        lime_results = []

    return {
        "shap": shap_values_row.tolist(),
        "lime": lime_results,
        "feature_names": feature_names
    }

def explain_lstm_kernel(model, X_seq, feature_names=None):
    samples, timesteps, features = X_seq.shape
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(features)]

    background = X_seq

    explainer = shap.KernelExplainer(
        lambda x: model.predict(x.reshape((-1, timesteps, features))).flatten(),
        background
    )

    shap_values = explainer.shap_values(X_seq[0:1])  # shape: (1, time_steps * features)

    try:
        shap_values = np.array(shap_values).reshape(timesteps, features)
        shap_avg = shap_values.mean(axis=0)
    except:
        shap_avg = np.mean(shap_values, axis=1).flatten()

    if len(shap_avg) != len(feature_names):
        shap_avg = shap_avg[:len(feature_names)]

    return {
        "shap": shap_avg.tolist(),
        "lime": [],
        "feature_names": feature_names
    }