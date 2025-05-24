from database.connection import SessionLocal
from database.models import Prediction
import json

def log_prediction(model, features, prob, explanation):
    db = SessionLocal()
    try:
        # Ensure all keys are present in explanation
        explanation_data = {
            "shap": explanation.get("shap", []),
            "lime": explanation.get("lime", []),
            "feature_names": explanation.get("feature_names", [f"f{i}" for i in range(len(explanation.get("shap", [])))])
        }

        pred = Prediction(
            model=model,
            features=json.dumps(features),
            fraud_prob=prob,
            explanation=json.dumps(explanation_data)
        )
        db.add(pred)
        db.commit()
    finally:
        db.close()
