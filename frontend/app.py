import streamlit as st
import pandas as pd
import sqlite3
import json
import ast
import matplotlib.pyplot as plt
import joblib
import os
import requests
from datetime import datetime, timezone, timedelta
from openai import OpenAI

# Initialize OpenAI client
client = None
if 'OPENAI_API_KEY' in st.secrets:
    client = OpenAI(api_key=st.secrets['OPENAI_API_KEY'])

st.set_page_config(page_title="AI Fraud Detection", layout="wide")
# Sidebar Navigation setup
page = st.sidebar.radio("🧭 Navigate", ["🧪 Manual Prediction", "🔍 Fraud Detection", "📈 Model Metrics", "📝 SAR Report"])

def format_time_ago(ts_unix):
    try:
        # Convert Unix timestamp to Helsinki time
        helsinki_tz = timezone(timedelta(hours=2))  # You might want to handle DST properly if needed
        file_time = datetime.fromtimestamp(ts_unix, tz=helsinki_tz)
        # Get current time in Helsinki
        now = datetime.now(helsinki_tz)

        delta = now - file_time
        days = delta.days
        hours, remainder = divmod(delta.seconds, 3600)
        minutes = remainder // 60

        parts = []
        if days > 0:
            parts.append(f"{days}d")
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0 or not parts:
            parts.append(f"{minutes}m")

        return " ".join(parts) + " ago"
    except Exception:
        return "unknown"


# Utility functions
def load_data():
    conn = sqlite3.connect("fraud.db")
    df = pd.read_sql_query("SELECT * FROM predictions ORDER BY timestamp DESC", conn)
    conn.close()
    return df

def parse_explanation(explanation_str):
    try:
        explanation = json.loads(explanation_str)
        shap_vals = explanation.get("shap", [])
        lime_vals = explanation.get("lime", [])
        feature_names = explanation.get("feature_names", [])
        return shap_vals, lime_vals, feature_names
    except Exception as e:
        st.warning(f"Failed to parse explanation: {e}")
        return [], [], []

# --- Page 1: Fraud Detection
if page == "🔍 Fraud Detection":
    st.title("🔍 AI Fraud Detection Dashboard")

    df = load_data()
    if df.empty:
        st.warning("No predictions logged yet.")
        st.stop()

    # Filter dropdown
    st.subheader("🧾 Prediction Table")
    filter_option = st.selectbox("🔍 Show:", ["All", "Only Fraud", "Only Non-Fraud"])
    filtered_df = df.copy()
    if filter_option == "Only Fraud":
        filtered_df = df[df["fraud_prob"] > 0.5]
    elif filter_option == "Only Non-Fraud":
        filtered_df = df[df["fraud_prob"] <= 0.5]

    # Display table
    display_df = filtered_df[["timestamp", "model", "fraud_prob"]].reset_index(drop=True)
    st.dataframe(display_df, use_container_width=True, height=400)

    # Row selector
    selected_row = st.selectbox(
        "📌 Select a row to inspect",
        display_df.index,
        format_func=lambda i: f"{display_df.loc[i, 'timestamp']} | {display_df.loc[i, 'model']}"
    )

    selected_full_row = filtered_df.iloc[selected_row]
    shap_vals, lime_vals, selected_features = parse_explanation(selected_full_row["explanation"])
    model_used = selected_full_row["model"]

    st.markdown("### 🧾 Feature Vector")
    try:
        raw_features = ast.literal_eval(selected_full_row["features"])
        if not selected_features:
            selected_features = joblib.load(f"models/features_{model_used}.pkl")
        feature_df = pd.DataFrame({
            "Feature": selected_features,
            "Value": raw_features[:len(selected_features)]
        })
        st.dataframe(feature_df, use_container_width=True)
    except Exception as e:
        st.error(f"⚠️ Failed to display features: {e}")

    st.subheader("📊 SHAP Feature Importance")
    if not selected_features:
        try:
            selected_features = joblib.load(f"models/features_{model_used}.pkl")
        except Exception as e:
            st.error(f"Error loading features: {e}")
            selected_features = [f"f{i}" for i in range(len(shap_vals))]

    if shap_vals:
        if len(shap_vals) != len(selected_features):
            st.error(f"⚠️ SHAP mismatch: Expected {len(selected_features)} features, got {len(shap_vals)}")
        else:
            shap_series = pd.Series(shap_vals, index=selected_features)
            top_n = 5
            top_features = shap_series.abs().nlargest(top_n).index
            st.write(f"Top {top_n} impactful features (by SHAP value):")
            fig, ax = plt.subplots()
            shap_series[top_features].plot(kind="barh", ax=ax, color="skyblue")
            ax.set_xlabel("SHAP value")
            ax.set_title("Feature Contribution to Fraud Probability")
            st.pyplot(fig)
    else:
        st.info("No SHAP values available.")

    st.subheader("🧠 LIME Explanation")
    if lime_vals:
        for rule, weight in lime_vals:
            st.write(f"- **{rule}** → contribution: {weight:.4f}")
    else:
        st.info("No LIME explanation available.")


# --- Page 2: Model Metrics
elif page == "📈 Model Metrics":
    st.title("📈 Model Training Logs & Management")

    st.subheader("🔁 Manual Retraining")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Retrain LightGBM"):
            try:
                res = requests.post("http://api:8000/retrain/lgbm")
                st.success(res.json().get("status", "Retrained"))
            except:
                st.error("❌ Failed to connect to API")
    with col2:
        if st.button("🔄 Retrain LSTM"):
            try:
                res = requests.post("http://api:8000/retrain/lstm")
                st.success(res.json().get("status", "Retrained"))
            except:
                st.error("❌ Failed to connect to API")

    st.subheader("🕒 Periodic Retraining")
    interval = st.number_input("Set retraining interval (minutes)", min_value=1, max_value=1440, value=60)
    col3, col4 = st.columns(2)
    with col3:
        if st.button("▶️ Start Periodic Retraining"):
            try:
                res = requests.post("http://api:8000/retrain/periodic", json={"interval": interval * 60})
                st.success(res.json().get("status", "Started"))
            except:
                st.error("Failed to start periodic retraining.")
    with col4:
        if st.button("⏹ Stop Periodic Retraining"):
            try:
                res = requests.post("http://api:8000/retrain/stop")
                st.success(res.json().get("status", "Stopped"))
            except:
                st.error("Failed to stop retraining.")

    st.subheader("📋 Current Models")

    for model_type in ["lightgbm", "lstm"]:
        model_path = f"models/{model_type}_model.pkl" if model_type == "lightgbm" else "models/lstm_model.h5"
        try:
            timestamp = os.path.getmtime(model_path)
            timestamp_str = pd.to_datetime(timestamp, unit='s').strftime('%Y-%m-%d %H:%M:%S')
            ago_str = format_time_ago(timestamp)
            st.markdown(f"**{model_type.upper()}** last updated: `{timestamp_str}` ({ago_str})")
        except:
            st.warning(f"⚠️ No saved model found for **{model_type.upper()}**.")

    st.subheader("📋 Training History")
    log_path = "results/training_logs.csv"
    if os.path.exists(log_path):
        logs = pd.read_csv(log_path)
        logs["timestamp"] = pd.to_datetime(logs["timestamp"], errors="coerce")
        logs = logs.sort_values(by="timestamp", ascending=False)
        st.dataframe(logs.reset_index(drop=True), use_container_width=True)
    else:
        st.info("No training logs found yet.")


# --- Page 4: SAR Report Generation
elif page == "📝 SAR Report":
    st.title("📝 Suspicious Activity Report (SAR) Generator")
    
    if client is None:
        st.error("⚠️ OpenAI API key not configured. Please add it to your secrets.")
        st.stop()
    
    # Load transaction data
    df = load_data()
    if df.empty:
        st.warning("No predictions logged yet.")
        st.stop()
    
    # Filter to show only suspicious transactions (fraud probability > 0.5)
    suspicious_df = df[df["fraud_prob"] > 0.5].copy()
    
    if suspicious_df.empty:
        st.info("No suspicious transactions found.")
        st.stop()
    
    # Transaction selector
    st.subheader("🔍 Select Transaction")
    selected_transaction = st.selectbox(
        "Select a suspicious transaction to generate SAR for:",
        suspicious_df.index,
        format_func=lambda i: f"Transaction from {suspicious_df.loc[i, 'timestamp']} (Fraud Prob: {suspicious_df.loc[i, 'fraud_prob']:.2f})"
    )
    
    selected_data = suspicious_df.loc[selected_transaction]
    
    # Display transaction details
    st.markdown("### 📊 Transaction Details")
    try:
        raw_features = ast.literal_eval(selected_data["features"])
        feature_names = joblib.load(f"models/features_{selected_data['model']}.pkl")
        feature_df = pd.DataFrame({
            "Feature": feature_names,
            "Value": raw_features[:len(feature_names)]
        })
        st.dataframe(feature_df, use_container_width=True)
    except Exception as e:
        st.error(f"⚠️ Failed to display transaction details: {e}")
    
    # Additional context input
    st.subheader("📝 Additional Context")
    additional_context = st.text_area(
        "Provide any additional context or observations about this transaction:",
        height=150
    )
    
    # Generate SAR draft
    if st.button("🤖 Generate SAR Draft"):
        try:
            # Prepare the prompt
            transaction_details = feature_df.to_string()
            model_explanation = f"Model used: {selected_data['model']}\nFraud Probability: {selected_data['fraud_prob']:.4f}"
            
            prompt = f"""Please generate a draft Suspicious Activity Report (SAR) based on the following information:

Transaction Details:
{transaction_details}

Model Assessment:
{model_explanation}

Additional Context:
{additional_context}

Please format the SAR draft with the following sections:
1. Subject Information
2. Suspicious Activity Information
3. Suspicious Activity Description
4. Additional Comments
"""
            
            # Call OpenAI API
            with st.spinner("🤖 Generating SAR draft..."):
                response = client.chat.completions.create(
                    model="gpt-4-turbo-preview",
                    messages=[
                        {"role": "system", "content": "You are a financial compliance expert specialized in writing Suspicious Activity Reports (SARs)."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7
                )
            
            # Display the generated SAR
            st.markdown("### 📄 Generated SAR Draft")
            st.markdown(response.choices[0].message.content)
            
            # Add download button
            st.download_button(
                label="📥 Download SAR Draft",
                data=response.choices[0].message.content,
                file_name=f"sar_draft_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
            
        except Exception as e:
            st.error(f"❌ Failed to generate SAR draft: {e}")

# --- Page 3: Manual Transaction Prediction
elif page == "🧪 Manual Prediction":
    st.title("🧪 Manual Transaction Submission")

    # --- LightGBM Section
    st.subheader("🔷 LightGBM: Paste One Row")

    try:
        features_lgbm = joblib.load("models/features_lgbm.pkl")
    except:
        st.error("❌ Failed to load features_lgbm.pkl")
        features_lgbm = []

    lgbm_input = st.text_area("Paste feature row (comma, space or tab separated)", height=100, key="lgbm_input")
    if st.button("🚀 Predict with LightGBM"):
        try:
            values = [float(x) for x in lgbm_input.replace(',', ' ').replace('\t', ' ').split()]
            selected = values[:len(features_lgbm)]
            while len(selected) < len(features_lgbm):
                selected.append(0.0)

            response = requests.post("http://api:8000/predict/lgbm", json={"features": selected})
            if response.status_code == 200:
                result = response.json()
                st.success(f"✅ LightGBM Fraud Probability: {result['fraud_probability']:.4f}")
            else:
                st.error(f"❌ Error: {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"❌ Parsing or request failed: {e}")

    st.markdown("---")

    # --- LSTM Section
    st.subheader("🔷 LSTM: Paste 3+ Rows for Sequence")

    try:
        features_lstm = joblib.load("models/features_lstm.pkl")
    except:
        st.error("❌ Failed to load features_lstm.pkl")
        features_lstm = []

    lstm_input = st.text_area("Paste 3 or more rows (one per line, each with 100+ values)", height=200, key="lstm_input")
    if st.button("🚀 Predict with LSTM"):
        try:
            rows = []
            for row in lstm_input.strip().split('\n'):
                if row.strip():
                    floats = [float(x) for x in row.replace(',', ' ').replace('\t', ' ').split()]
                    rows.append(floats)

            if len(rows) < 3:
                st.error("❌ Please paste at least 3 rows.")
            else:
                sequence = []
                for row in rows[:3]:
                    trimmed = row[:len(features_lstm)]
                    while len(trimmed) < len(features_lstm):
                        trimmed.append(0.0)
                    sequence.append(trimmed)

                response = requests.post("http://api:8000/predict/lstm", json={"features": sequence})
                if response.status_code == 200:
                    result = response.json()
                    st.success(f"✅ LSTM Fraud Probability: {result['fraud_probability']:.4f}")
                else:
                    st.error(f"❌ Error: {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"❌ Parsing or request failed: {e}")
