import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model_api.shap_lime import explain_with_shap_and_lime
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def test_explain_with_shap_and_lime():
    # 1. Mock data
    X = pd.DataFrame(np.random.rand(20, 4), columns=["f1", "f2", "f3", "f4"])
    y = np.random.randint(0, 2, 20)

    # 2. Train a dummy model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)

    # 3. Run explanation function
    result = explain_with_shap_and_lime(model, X, feature_names=list(X.columns))

    # 4. Assertions
    assert "shap" in result
    assert "lime" in result
    assert "feature_names" in result
    assert isinstance(result["shap"], list)
    assert isinstance(result["lime"], list)
    assert len(result["feature_names"]) == 4
    # Optional: check SHAP explanation length
    assert len(result["shap"]) == 4

    print("SHAP values:", result["shap"])
    print("LIME results:", result["lime"])

