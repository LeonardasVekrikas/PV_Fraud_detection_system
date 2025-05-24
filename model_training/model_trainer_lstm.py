# model_training/model_trainer_lstm.py
import numpy as np
import pandas as pd
import joblib
import os
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, matthews_corrcoef,
    recall_score, precision_score, f1_score, roc_curve, auc, RocCurveDisplay
)
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import SGDOneClassSVM
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense

def train_lstm_balanced(X, y, model_path='models/lstm_model.h5', n_splits=5):
    # Create windowed sequences BEFORE splitting
    w = 3
    X_np = X.to_numpy()
    y_np = y.to_numpy()

    X_seq = np.array([X_np[i:i + w] for i in range(len(X_np) - w + 1)])
    y_seq = y_np[w - 1:]

    tscv = TimeSeriesSplit(n_splits=n_splits)
    f1, acc, recall, precision, AUC, mcc, bal_acc = [], [], [], [], [], [], []
    tprs, aucs = [], []
    mean_fpr = np.linspace(0, 1, 100)
    fig, ax = plt.subplots(figsize=(6, 6))

    k = 0
    for train_index, test_index in tscv.split(X_seq):
        X_train_seq, X_test_seq = X_seq[train_index], X_seq[test_index]
        y_train_seq, y_test_seq = y_seq[train_index], y_seq[test_index]

        # Flatten and normalize training data
        flat_train = X_train_seq.reshape(-1, X.shape[1])
        scaler = StandardScaler()
        flat_train = scaler.fit_transform(flat_train)
        X_train_seq = flat_train.reshape(X_train_seq.shape)

        flat_test = X_test_seq.reshape(-1, X.shape[1])
        flat_test = scaler.transform(flat_test)
        X_test_seq = flat_test.reshape(X_test_seq.shape)

        # Balance using OneClassSVM + RUS on flattened sequence start points
        normal_train_flat = X_train_seq[y_train_seq == 0, -1, :]
        clf = SGDOneClassSVM(nu=0.025, random_state=0)
        res = clf.fit_predict(normal_train_flat)
        res = np.where(res == 1, 0, 1)
        if len(np.unique(res)) < 2:
            print(f"[WARNING] OneClassSVM returned one class on fold {k}. Skipping resampling.")
        else:
            rus = RandomUnderSampler(sampling_strategy=0.95)
            _, _ = rus.fit_resample(normal_train_flat, res)
            drop_idx = rus.sample_indices_
            keep_mask = np.ones(len(X_train_seq), dtype=bool)
            normal_indices = np.where(y_train_seq == 0)[0]
            drop_global = normal_indices[drop_idx]
            keep_mask[drop_global] = False
            X_train_seq = X_train_seq[keep_mask]
            y_train_seq = y_train_seq[keep_mask]

        # Model
        inputs = Input(shape=(w, X.shape[1]))
        x = LSTM(50)(inputs)
        outputs = Dense(1, activation='sigmoid')(x)
        model = Model(inputs, outputs)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['Precision', 'Recall'])

        model.fit(X_train_seq, y_train_seq, epochs=10, batch_size=512, verbose=0)
        y_pred = model.predict(X_test_seq)[:, 0] > 0.5
        y_pred = y_pred.astype(int)

        fpr, tpr, _ = roc_curve(y_test_seq, y_pred)
        roc_display = RocCurveDisplay.from_predictions(y_test_seq, y_pred, name=f"ROC fold {k}", alpha=0.3, lw=1, ax=ax)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(roc_display.roc_auc)

        acc.append(accuracy_score(y_test_seq, y_pred))
        precision.append(precision_score(y_test_seq, y_pred))
        recall.append(recall_score(y_test_seq, y_pred))
        f1.append(f1_score(y_test_seq, y_pred))
        AUC.append(roc_display.roc_auc)
        mcc.append(matthews_corrcoef(y_test_seq, y_pred))
        bal_acc.append(balanced_accuracy_score(y_test_seq, y_pred))
        k += 1

    print("Mean ACC:", np.mean(acc), "Precision:", np.mean(precision), "Recall:", np.mean(recall),
          "F1:", np.mean(f1), "AUC:", np.mean(AUC), "MCC:", np.mean(mcc), "Balanced Acc:", np.mean(bal_acc))

    ax.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color="b", label=f"Mean ROC (AUC = {mean_auc:.2f} ± {std_auc:.2f})", lw=2, alpha=0.8)
    ax.fill_between(mean_fpr, np.maximum(mean_tpr - np.std(tprs, axis=0), 0), np.minimum(mean_tpr + np.std(tprs, axis=0), 1),
                    color="grey", alpha=0.2, label="± 1 std. dev.")
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], xlabel="False Positive Rate", ylabel="True Positive Rate",
           title="Mean ROC curve\n(Sequential-European dataset)")
    ax.axis("square")
    ax.legend(loc="lower right")
    plt.show()

    os.makedirs("models", exist_ok=True)
    model.save(model_path)
    joblib.dump(X.columns.tolist(), "models/features_lstm.pkl")

    os.makedirs("results", exist_ok=True)
    log_entry = pd.DataFrame([{
        "timestamp": datetime.now().isoformat(),
        "model": "lstm",
        "accuracy": np.mean(acc),
        "recall": np.mean(recall),
        "precision": np.mean(precision),
        "f1_score": np.mean(f1),
        "auc": np.mean(AUC)
    }])
    log_path = "results/training_logs.csv"
    if os.path.exists(log_path):
        log_entry.to_csv(log_path, mode='a', header=False, index=False)
    else:
        log_entry.to_csv(log_path, index=False)

    print("[SUCCESS] LSTM model and features saved.")
    return model
