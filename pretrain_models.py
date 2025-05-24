import pandas as pd
import joblib
from model_training.model_trainer_lightgbm import train_lightgbm_balanced
from model_training.model_trainer_lstm import train_lstm_balanced

DATA_PATH = r"C:\Users\strei\Documents\!Magistrantūra\MBD\Kodas\Fraud-docker copy\data\creditcard.csv"

def main():
    print("[INFO] Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    if "Class" not in df.columns:
        raise ValueError("Dataset must contain a 'Class' column.")

    print("[INFO] Training LightGBM...")
    train_lightgbm_balanced(df)

    print("[INFO] Preparing feature set for LSTM...")
    try:
        selected_features = joblib.load("models/features_lgbm.pkl")
    except FileNotFoundError:
        raise RuntimeError("❌ Cannot train LSTM: LightGBM features file not found.")

    X = df[selected_features]
    y = df["Class"]

    print("[INFO] Training LSTM...")
    train_lstm_balanced(X, y)

    print("[✅ DONE] Both models trained and saved.")

if __name__ == "__main__":
    main()
