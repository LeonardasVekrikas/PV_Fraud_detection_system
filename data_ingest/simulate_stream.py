import requests
import pandas as pd
import time

df = pd.read_csv("creditcard.csv")

for _, row in df.iterrows():
    features = row.drop("Class").tolist()
    response = requests.post("http://localhost:8000/predict/lgbm", json={"features": features})
    print(response.json())
    time.sleep(1)
