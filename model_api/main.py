from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import pandas as pd
from model_api.lightgbm_model import predict_lgbm
from model_api.lstm_model import predict_lstm
from model_training.model_trainer_lightgbm import train_lightgbm_balanced
from model_training.model_trainer_lstm import train_lstm_balanced
from database.crud import log_prediction
from database.connection import engine
from database.models import Base
import threading
import time
import os
from fastapi.responses import JSONResponse

Base.metadata.create_all(bind=engine)

app = FastAPI()

class Transaction(BaseModel):
    features: list

periodic_retraining = {
    "enabled": False,
    "interval": 3600  # seconds
}

def retrain_periodically():
    print(f"[INFO] Background retraining running every {periodic_retraining['interval']}s")
    while periodic_retraining["enabled"]:
        try:
            print("[INFO] Periodic retraining started")
            df = pd.read_csv("data/creditcard.csv")
            train_lightgbm_balanced(df)
            train_lstm_balanced(df.drop(columns=['Class']), df['Class'])
            print("[INFO] Periodic retraining completed")
        except Exception as e:
            print("[ERROR] Periodic retraining failed:", e)
        time.sleep(periodic_retraining["interval"])

@app.get("/logs/training")
def get_training_logs():
    log_path = "results/training_logs.csv"
    try:
        if os.path.exists(log_path):
            df = pd.read_csv(log_path)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp", ascending=False)
            return df.to_dict(orient="records")
        else:
            return JSONResponse(content={"message": "No training logs found."}, status_code=404)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/predict/lgbm")
def predict_lightgbm(transaction: Transaction):
    try:
        prob, explanation = predict_lgbm(transaction.features)
        log_prediction("lgbm", transaction.features, prob, explanation)
        return {"model": "lgbm", "fraud_probability": prob, "explanation": explanation}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/lstm")
def predict_lstm_model(transaction: Transaction):
    try:
        prob, explanation = predict_lstm(transaction.features)
        log_prediction("lstm", transaction.features, prob, explanation)
        return {"model": "lstm", "fraud_probability": prob, "explanation": explanation}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/retrain/lgbm")
def retrain_lgbm():
    try:
        df = pd.read_csv("data/creditcard.csv")
        train_lightgbm_balanced(df)
        return {"status": "LightGBM model retrained successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/retrain/lstm")
def retrain_lstm():
    try:
        df = pd.read_csv("data/creditcard.csv")
        train_lstm_balanced(df.drop(columns=['Class']), df['Class'])
        return {"status": "LSTM model retrained successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/retrain/periodic")
async def start_periodic_retraining(request: Request):
    try:
        data = await request.json()
        interval = data.get("interval", 3600)
        periodic_retraining["interval"] = int(interval)
        if not periodic_retraining["enabled"]:
            periodic_retraining["enabled"] = True
            thread = threading.Thread(target=retrain_periodically, daemon=True)
            thread.start()
        return {"status": f"Periodic retraining started every {interval} seconds"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/retrain/stop")
def stop_periodic_retraining():
    periodic_retraining["enabled"] = False
    return {"status": "Periodic retraining stopped"}
