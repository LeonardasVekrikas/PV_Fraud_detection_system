# model_training/model_trainer_lightgbm.py
import joblib
import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.svm import OneClassSVM
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix,
    accuracy_score, recall_score, precision_score, f1_score
)

# patobulintas lightgbm treniravimas
def train_lightgbm_balanced(df, target_column='Class', model_path='models/lightgbm_model.pkl'):
    print("[INFO] Splitting normal and fraud classes...")
    df_fraud = df[df[target_column] == 1].copy()
    df_normal = df[df[target_column] == 0].copy()

    print("[INFO] Applying One-Class SVM...")
    clf_osvm = OneClassSVM()
    df_fraud['OSVM_Label'] = clf_osvm.fit_predict(df_fraud.drop(columns=[target_column]))
    osvm_res = df_fraud[df_fraud['OSVM_Label'] != -1].drop(columns=['OSVM_Label'])
    osvm_out = df_fraud[df_fraud['OSVM_Label'] == -1].drop(columns=['OSVM_Label'])

    y_inliers = pd.Series([1] * len(osvm_res), name='Class')
    y_outliers = pd.Series([1] * len(osvm_out), name='Class')

    df_normal[target_column] = 0
    df_combined = pd.concat([df_normal, osvm_res], ignore_index=True)
    y_combined = df_combined[target_column]
    X_combined = df_combined.drop(columns=[target_column])

    print("[INFO] Applying SMOTE...")
    smote = SMOTE(random_state=0, sampling_strategy=0.0526)
    X_resampled, y_resampled = smote.fit_resample(X_combined, y_combined)

    X_final = pd.concat([X_resampled, osvm_out], ignore_index=True)
    y_final = pd.concat([y_resampled, y_outliers], ignore_index=True)

    print("[INFO] Splitting into train and validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_final, y_final, test_size=0.2, stratify=y_final, random_state=42
    )

    print("[INFO] Training LightGBM...")
    clf = LGBMClassifier(
        colsample_bytree=0.7, is_unbalance=False, learning_rate=0.01,
        num_iterations=600, max_bin=100, max_depth=16, metric='f1',
        min_child_samples=100, min_child_weight=0, n_estimators=5000,
        num_leaves=1000, random_state=0, subsample_freq=0, verbose=-1,
        reg_alpha=0.5
    )

    clf.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='auc'
    )

    print("[METRICS] LightGBM Validation Metrics:")
    y_pred = clf.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_pred)

    print(confusion_matrix(y_val, y_pred))
    print(classification_report(y_val, y_pred, digits=4))
    print(f"AUC: {auc:.4f}")

    os.makedirs("models", exist_ok=True)
    joblib.dump(clf, model_path)
    joblib.dump(X_train.columns.tolist(), "models/features_lgbm.pkl")

    os.makedirs("results", exist_ok=True)
    log_entry = pd.DataFrame([{
        "timestamp": datetime.now().isoformat(),
        "model": "lightgbm",
        "accuracy": acc,
        "recall": recall,
        "precision": precision,
        "f1_score": f1,
        "auc": auc
    }])
    log_path = "results/training_logs.csv"
    if os.path.exists(log_path):
        log_entry.to_csv(log_path, mode='a', header=False, index=False)
    else:
        log_entry.to_csv(log_path, index=False)

    print("[SUCCESS] LightGBM model and features saved.")
    return clf