import sys
import os

# --- Force project root onto sys.path as the first import location ---
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

from model_api.main import app

client = TestClient(app)

# --- Minimal test to ensure pytest can discover and run tests ---
def test_pytest_discovery():
    assert True

# --- Test: No training logs found ---
def test_get_training_logs_no_logs(monkeypatch):
    monkeypatch.setattr("os.path.exists", lambda x: False)
    response = client.get("/logs/training")
    assert response.status_code == 404
    assert response.json()["message"] == "No training logs found."

# --- Test: LGBM prediction endpoint ---
@patch("model_api.main.predict_lgbm")
@patch("model_api.main.log_prediction")
def test_predict_lgbm(mock_log_prediction, mock_predict_lgbm):
    mock_predict_lgbm.return_value = (0.25, {"reason": "Mocked explanation"})
    payload = {"features": [0.5] * 31}
    response = client.post("/predict/lgbm", json=payload)
    print(response.text)
    assert response.status_code == 200
    res = response.json()
    assert res["model"] == "lgbm"
    assert "fraud_probability" in res
    assert "explanation" in res
    mock_log_prediction.assert_called_once()

# --- Test: LSTM prediction endpoint ---
@patch("model_api.main.predict_lstm")
@patch("model_api.main.log_prediction")
def test_predict_lstm(mock_log_prediction, mock_predict_lstm):
    mock_predict_lstm.return_value = (0.99, {"reason": "Mocked explanation"})
    payload = {"features": [0.1] * 15}
    response = client.post("/predict/lstm", json=payload)
    print(response.text)
    assert response.status_code == 200
    res = response.json()
    assert res["model"] == "lstm"
    assert "fraud_probability" in res
    assert "explanation" in res
    mock_log_prediction.assert_called_once()

# --- Test: LightGBM retraining endpoint ---
@patch("model_api.main.train_lightgbm_balanced")
@patch("model_api.main.pd.read_csv")
def test_retrain_lgbm(mock_read_csv, mock_train_lightgbm):
    import pandas as pd
    mock_read_csv.return_value = pd.DataFrame({"A": [1, 2], "Class": [0, 1]})
    mock_train_lightgbm.return_value = None
    response = client.post("/retrain/lgbm")
    print(response.text)
    assert response.status_code == 200
    assert response.json()["status"].startswith("LightGBM model retrained")

# --- Test: LSTM retraining endpoint ---
@patch("model_api.main.train_lstm_balanced")
@patch("model_api.main.pd.read_csv")
def test_retrain_lstm(mock_read_csv, mock_train_lstm):
    import pandas as pd
    mock_read_csv.return_value = pd.DataFrame({"A": [1, 2], "Class": [0, 1]})
    mock_train_lstm.return_value = None
    response = client.post("/retrain/lstm")
    print(response.text)
    assert response.status_code == 200
    assert response.json()["status"].startswith("LSTM model retrained")

# --- Test: Periodic retraining start/stop ---
def test_start_and_stop_periodic_retraining():
    response = client.post("/retrain/periodic", json={"interval": 2})
    assert response.status_code == 200
    assert "Periodic retraining started" in response.json()["status"]

    response = client.post("/retrain/stop")
    assert response.status_code == 200
    assert response.json()["status"] == "Periodic retraining stopped"

# --- OPTIONAL: Negative test, invalid payload for predict ---
def test_predict_lgbm_invalid_payload():
    payload = {"not_features": [0.5] * 31}
    response = client.post("/predict/lgbm", json=payload)
    assert response.status_code == 422  # Unprocessable Entity

# --- OPTIONAL: Negative test, invalid endpoint ---
def test_nonexistent_endpoint():
    response = client.get("/doesnotexist")
    assert response.status_code == 404
