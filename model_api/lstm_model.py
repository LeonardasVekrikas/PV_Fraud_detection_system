import joblib
import numpy as np
from tensorflow.keras.models import load_model
from model_api.shap_lime import explain_lstm_kernel

model = load_model("models/lstm_model.h5")
feature_names = joblib.load("models/features_lstm.pkl")

def predict_lstm(features):
    import numpy as np
    n_features = len(feature_names)

    # Convert to numpy array
    X_seq = np.array(features, dtype=np.float32)

    # Validate shape
    if X_seq.ndim == 2 and X_seq.shape[0] == 3 and X_seq.shape[1] == n_features:
        # shape: (3, n_features) → add batch dim
        X_seq = X_seq.reshape((1, 3, n_features))
    elif X_seq.ndim == 3 and X_seq.shape[1:] == (3, n_features):
        # shape is already (1, 3, n_features)
        pass
    else:
        raise ValueError(f"Invalid input shape for LSTM model: got {X_seq.shape}, expected (3, {n_features}) or (1, 3, {n_features})")

    # Predict
    prob = model.predict(X_seq)[0][0]
    explanation = explain_lstm_kernel(model, X_seq, feature_names)
    return prob, explanation
